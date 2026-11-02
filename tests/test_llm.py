import json

import httpx
import pytest

from asr_dashboard.config.schema import AppConfig
from asr_dashboard.models import Segment, TranscriptionResult
from asr_dashboard.text import llm_postprocess as LP
from asr_dashboard.text.llm_client import LLMClient, is_local, validate
from asr_dashboard.text.postprocess import mode_label, run_postprocess, select_mode, undo_all, undo_segment


def reply(content, status=200):
    return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})


def tx(handler):
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------- validation
def test_validate_network_error():
    def h(req):
        raise httpx.ConnectError("refused", request=req)

    r = validate("http://localhost:9/v1", "", "m", transport=tx(h))
    assert not r.ok and r.kind == "network" and "地址" in r.hint


def test_validate_auth_and_model_and_parse():
    r = validate("https://x/v1", "bad", "m", transport=tx(lambda q: httpx.Response(401, text="invalid api key")))
    assert not r.ok and r.kind == "auth"
    r = validate("https://x/v1", "k", "nope", transport=tx(lambda q: httpx.Response(
        400, json={"error": {"message": "The model `nope` does not exist"}})))
    assert not r.ok and r.kind == "model"
    r = validate("https://x/v1", "k", "m", transport=tx(lambda q: reply("I am fine thanks")))
    assert not r.ok and r.kind == "parse"
    r = validate("https://x/v1", "k", "m", transport=tx(lambda q: httpx.Response(200, text="<html>")))
    assert not r.ok and r.kind == "parse"


def test_validate_success_and_json_mode_detection():
    seen = []

    def h(req):
        body = json.loads(req.content)
        seen.append("response_format" in body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format json_object is not supported"}})
        return reply('```json\n{"ok": true}\n```')

    r = validate("http://localhost:11434/v1", "", "qwen3:8b", transport=tx(h))
    assert r.ok and r.json_mode is False and seen == [True, False]
    r2 = validate("https://x/v1", "k", "m", transport=tx(lambda q: reply('{"ok": true}')))
    assert r2.ok and r2.json_mode is True
    assert is_local("http://127.0.0.1:1234/v1") and not is_local("https://api.deepseek.com/v1")


# ---------------------------------------------------------------- mode selection
def test_mode_selection():
    c = AppConfig()
    assert select_mode(c) == "rules"  # unconfigured
    c.llm.base_url, c.llm.model, c.llm.validated = "https://x/v1", "m", True
    c.llm.enabled = False
    assert select_mode(c) == "rules"  # configured but switched off
    c.llm.enabled = True
    assert select_mode(c) == "llm"  # configured + validated + on
    c.llm.validated = False
    assert select_mode(c) == "rules"


# ---------------------------------------------------------------- cleanup
def result():
    segs = [Segment(f"第{i}句 嗯 的的内容", i, i + 1, "SPEAKER_00") for i in range(6)]
    return TranscriptionResult("f.wav", segs, "", "zh", 6)


def cfg_llm(chunk_chars=200):
    c = AppConfig()
    c.llm.base_url, c.llm.model, c.llm.validated = "https://x/v1", "deepseek-chat", True
    c.llm.chunk_chars = chunk_chars
    c.llm.concurrency = 1
    return c


def fixer(mutate=None):
    def h(req):
        payload = json.loads(json.loads(req.content)["messages"][1]["content"])
        items = [{"id": it["id"], "text": it["text"].replace(" 嗯 ", "").replace("的的", "的")} for it in payload["items"]]
        if mutate:
            items = mutate(payload, items)
        return reply(json.dumps({"items": items}, ensure_ascii=False))
    return h


def run(cfg, handler):
    r = result()
    client = LLMClient(cfg.llm.base_url, "k", cfg.llm.model, transport=tx(handler))
    run_postprocess(r, cfg, "k", "术语：千问", client=client)
    return r


def test_llm_cleanup_success_and_undo():
    r = run(cfg_llm(), fixer())
    assert r.postprocess_mode == "llm" and mode_label(r, cfg_llm()) == "大模型：deepseek-chat"
    assert r.segments[0].text == "第0句的内容" and r.segments[0].original_text == "第0句 嗯 的的内容"
    assert len(r.segments) == 6  # no merges in LLM mode
    undo_segment(r, 0)
    assert r.segments[0].text == "第0句 嗯 的的内容" and r.segments[0].original_text is None
    assert undo_all(r) == 5 and all(s.original_text is None for s in r.segments)


def test_llm_id_mismatch_falls_back_per_chunk():
    def mutate(payload, items):
        return items[:-1] if payload["items"][0]["id"] == 0 else items

    r = run(cfg_llm(), fixer(mutate))
    assert r.postprocess_mode == "llm+rules"
    assert any("id 与输入不一致" in w for w in r.warnings)
    assert mode_label(r, cfg_llm()).startswith("大模型：deepseek-chat")


def test_llm_change_ratio_and_broken_json():
    r = run(cfg_llm(10_000), fixer(lambda p, items: [{"id": i["id"], "text": "完全不同的一段新内容新内容新内容"} for i in items]))
    assert r.postprocess_mode == "rules" and any("改动比例" in w for w in r.warnings)
    r = run(cfg_llm(10_000), lambda req: reply("{not json"))
    assert r.postprocess_mode == "rules" and any("JSON" in w for w in r.warnings)


def test_previous_context_is_readonly_and_chunked():
    seen = []

    def h(req):
        payload = json.loads(json.loads(req.content)["messages"][1]["content"])
        seen.append(payload)
        return fixer()(req)

    run(cfg_llm(200), h)
    assert len(seen) >= 2 and seen[0]["context"] == "术语：千问"
    assert seen[1]["previous"] and seen[1]["previous"][-1]["id"] == seen[1]["items"][0]["id"] - 1


def test_change_ratio_metric():
    assert LP.change_ratio("abc", "abc") == 0
    assert LP.change_ratio("abcd", "abxd") == pytest.approx(0.25)


def test_summary_and_speaker_guess():
    c = cfg_llm()
    c.llm.tasks.summary = True
    c.llm.tasks.speaker_guess = True
    r = result()
    from asr_dashboard.models import SpeakerInfo

    r.speakers = {"SPEAKER_00": SpeakerInfo("SPEAKER_00"), "SPEAKER_01": SpeakerInfo("SPEAKER_01", "李四", "p1", 0.9)}

    def h(req):
        sys_prompt = json.loads(req.content)["messages"][0]["content"]
        if "摘要" in sys_prompt:
            return reply("## 要点\n- a")
        if "推测" in sys_prompt:
            return reply(json.dumps({"suggestions": [{"speaker": "SPEAKER_00", "name": "王总", "reason": "被称呼王总"},
                                                     {"speaker": "SPEAKER_01", "name": "x", "reason": "y"}]}))
        return fixer()(req)

    client = LLMClient(c.llm.base_url, "k", c.llm.model, transport=tx(h))
    run_postprocess(r, c, "k", "", client=client)
    assert r.summary.startswith("## 要点")
    assert r.speaker_suggestions == [{"speaker": "SPEAKER_00", "name": "王总", "reason": "被称呼王总"}]

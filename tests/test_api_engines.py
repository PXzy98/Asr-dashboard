import json
import shutil

import httpx
import numpy as np
import pytest

from asr_dashboard.config.schema import ApiModelConfig
from asr_dashboard.engines.api import create_api_engine, from_preset
from asr_dashboard.engines.api.base import APIError, parse_retry_after
from asr_dashboard.engines.api.custom_http import extract, render
from asr_dashboard.engines.api.openai_chat_audio import parse_asr_output
from asr_dashboard.models import AudioChunk, CancelledError, CancelToken

KEY = "sk-test-0123456789abcdef"


def chunk(sec=5.0, off=0.0):
    rng = np.random.default_rng(0)
    return AudioChunk(off, sec, samples=(0.1 * rng.standard_normal(int(sec * 16000))).astype("float32"))


def engine(handler, **cfg):
    c = ApiModelConfig(**{"name": "t", "base_url": "https://api.example.com/v1", "model": "m", **cfg})
    sleeps = []
    e = create_api_engine(c, KEY, transport=httpx.MockTransport(handler))
    e._sleep = sleeps.append
    return e, sleeps


def verbose(text="hello world.", start=0.5, end=2.0):
    return {"text": text, "language": "english",
            "segments": [{"start": start, "end": end, "text": text}],
            "words": [{"word": "hello", "start": 0.5, "end": 1.0}, {"word": "world", "start": 1.1, "end": 2.0}]}


def test_retry_429_with_retry_after_then_success():
    calls = []

    def h(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="slow down")
        if len(calls) == 2:
            return httpx.Response(500, text="oops")
        return httpx.Response(200, json=verbose())

    e, sleeps = engine(h, retries=3)
    segs = e.transcribe([chunk(off=10)], None, "术语", lambda *a: None, CancelToken())
    assert len(calls) == 3 and sleeps == [7.0, 2.0]
    assert segs[0].start == pytest.approx(10.5) and segs[0].end == pytest.approx(12.0)
    assert [w.text for w in segs[0].words] == ["hello", "world"] and segs[0].words[0].start == pytest.approx(10.5)
    body = calls[-1].content.decode("latin-1")
    assert 'name="prompt"' in body and 'name="response_format"' in body and "verbose_json" in body
    assert calls[-1].headers["authorization"] == f"Bearer {KEY}"


def test_timeout_retry_and_exhaustion_redacts_key():
    def h(req):
        raise httpx.ReadTimeout(f"timed out for key {KEY}", request=req)

    e, sleeps = engine(h, retries=2)
    with pytest.raises(APIError) as ei:
        e.transcribe([chunk()], None, None, lambda *a: None, CancelToken())
    assert ei.value.kind == "timeout" and len(sleeps) == 2
    assert KEY not in str(ei.value)


def test_non_retryable_auth_error():
    e, sleeps = engine(lambda r: httpx.Response(401, text="bad key"), retries=3)
    with pytest.raises(APIError) as ei:
        e.transcribe([chunk()], None, None, lambda *a: None, CancelToken())
    assert ei.value.kind == "auth" and sleeps == []


def test_verbose_json_fallback_to_json():
    seen = []

    def h(req):
        body = req.content.decode("latin-1")
        seen.append("verbose_json" in body)
        if "verbose_json" in body:
            return httpx.Response(400, json={"error": {"message": "response_format verbose_json not supported"}})
        return httpx.Response(200, json={"text": "plain text"})

    e, _ = engine(h)
    segs = e.transcribe([chunk(4, off=3)], None, None, lambda *a: None, CancelToken())
    assert seen == [True, False]
    assert segs[0].text == "plain text" and (segs[0].start, segs[0].end) == (3.0, 7.0)
    assert not e.capabilities.segment_timestamps and e.wants_chunked_input()


def test_retry_after_parse():
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg")
def test_size_limit_splits_and_reassembles():
    # 120 s of noise = ~3.8 MB wav; limit 0.5 MB forces compression and splitting
    sizes = []

    def h(req):
        body = req.content
        sizes.append(len(body))
        return httpx.Response(200, json={"text": f"part{len(sizes)}"})

    e, _ = engine(h, max_file_mb=0.5, timestamps=False, concurrency=3)
    c = chunk(120.0)
    ups = e.prepare_uploads(c)
    assert len(ups) >= 2 and all(len(u.data) <= 0.5 * 1024 * 1024 for u in ups)
    assert ups[0].offset == 0 and abs(sum(u.duration for u in ups) - 120) < 0.01
    segs = e.transcribe([c], None, None, lambda *a: None, CancelToken())
    assert len(segs) == len(ups) and [s.start for s in segs] == sorted(s.start for s in segs)
    assert segs[-1].end == pytest.approx(120, abs=0.05)


def test_max_audio_seconds_split():
    e, _ = engine(lambda r: httpx.Response(200, json=verbose("hi.", 0.1, 0.5)), max_audio_seconds=30)
    ups = e.prepare_uploads(chunk(70.0, off=100))
    assert all(u.duration <= 30 + 1e-6 for u in ups) and ups[0].offset == 100


def test_cancel_aborts_inflight():
    import threading

    tok = CancelToken()
    started = threading.Event()

    def h(req):
        started.set()
        tok.wait(5)
        raise httpx.ReadError("closed", request=req)

    e, _ = engine(h, retries=5)
    threading.Timer(0.2, tok.cancel).start()
    with pytest.raises(CancelledError):
        e.transcribe([chunk()], None, None, lambda *a: None, tok)


def test_chat_audio_parse_and_request():
    assert parse_asr_output("language Chinese<asr_text>你好") == ("Chinese", "你好")
    assert parse_asr_output("language None<asr_text>") == (None, "")
    assert parse_asr_output("plain") == (None, "plain")
    got = {}

    def h(req):
        got.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "language English<asr_text>hi there"}}]})

    e, _ = engine(h, type="openai_chat_audio", timestamps=False)
    segs = e.transcribe([chunk(3, off=5)], None, "context words", lambda *a: None, CancelToken())
    assert got["messages"][0] == {"role": "system", "content": "context words"}
    assert got["messages"][1]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert segs[0].text == "hi there" and segs[0].language == "en" and segs[0].start == 5


def test_custom_http_jsonpath_sync():
    o = {"text_path": "$.result.text", "segments_path": "$.result.items", "seg_start": "$.b", "seg_end": "$.e",
         "seg_text": "$.t", "seg_speaker": "$.who", "time_unit": "ms"}
    data = {"result": {"text": "a b", "items": [{"b": 100, "e": 900, "t": "a", "who": 1}, {"b": 1000, "e": 1500, "t": "b"}]}}
    text, segs = extract(data, o)
    assert text == "a b" and segs[0].start == pytest.approx(0.1) and segs[0].speaker == "1" and segs[1].end == 1.5
    assert render({"k": "{{model}}", "n": "{{x}}", "s": "pre-{{model}}"}, {"model": "M", "x": 3}) == \
        {"k": "M", "n": 3, "s": "pre-M"}


def test_custom_http_async_polling():
    polls = []

    def h(req):
        if req.method == "POST":
            body = json.loads(req.content)
            assert body["audio"] and body["lang"] == "zh" and req.headers["x-key"] == KEY
            return httpx.Response(200, json={"task": {"id": "T1"}})
        polls.append(str(req.url))
        if len(polls) < 3:
            return httpx.Response(200, json={"state": "running"})
        return httpx.Response(200, json={"state": "ok", "data": {"text": "done!", "segs": [{"start": 0.2, "end": 1.2, "text": "done!"}]}})

    opts = {"url": "{{base_url}}/submit", "headers": {"X-Key": "{{api_key}}"}, "body_type": "json_base64",
            "body": {"audio": "{{audio_base64}}", "lang": "{{language}}"}, "async": True,
            "task_id_path": "$.task.id", "poll_url": "{{base_url}}/tasks/{{task_id}}", "status_path": "$.state",
            "done_values": "ok", "failed_values": "bad", "result_path": "$.data", "text_path": "$.text",
            "segments_path": "$.segs"}
    e, sleeps = engine(h, type="custom_http", options=opts)
    segs = e.transcribe([chunk(2, off=1)], "zh", None, lambda *a: None, CancelToken())
    assert polls[-1].endswith("/tasks/T1") and len(sleeps) == 2
    assert segs[0].text == "done!" and segs[0].start == pytest.approx(1.2)


def test_dashscope_flow():
    seen = []

    def h(req):
        u = str(req.url)
        seen.append((req.method, u, dict(req.headers)))
        if "/uploads" in u:
            return httpx.Response(200, json={"data": {"policy": "p", "signature": "s", "upload_dir": "dir",
                                                      "upload_host": "https://oss.example.com", "oss_access_key_id": "ak"}})
        if u.startswith("https://oss.example.com"):
            return httpx.Response(200)
        if u.endswith("/services/audio/asr/transcription"):
            b = json.loads(req.content)
            assert b["input"]["file_url"].startswith("oss://dir/")
            return httpx.Response(200, json={"output": {"task_id": "abc", "task_status": "PENDING"}})
        if "/tasks/abc" in u:
            st = "RUNNING" if sum("/tasks/" in s[1] for s in seen) < 2 else "SUCCEEDED"
            return httpx.Response(200, json={"output": {"task_status": st, "result": {"transcription_url": "https://res.example.com/r.json"}}})
        if u.startswith("https://res.example.com"):
            return httpx.Response(200, json={"transcripts": [{"text": "你好。再见。", "sentences": [
                {"begin_time": 100, "end_time": 900, "text": "你好。"}, {"begin_time": 1200, "end_time": 2000, "text": "再见。"}]}]})
        return httpx.Response(404)

    e, _ = engine(h, type="dashscope_qwen", base_url="", model="qwen3-asr-flash-filetrans")
    segs = e.transcribe([chunk(3)], None, None, lambda *a: None, CancelToken())
    assert [s.text for s in segs] == ["你好。", "再见。"] and segs[1].start == pytest.approx(1.2)
    submit = next(s for s in seen if s[1].endswith("/transcription"))
    assert submit[2]["x-dashscope-async"] == "enable" and submit[2]["x-dashscope-ossresourceresolve"] == "enable"
    assert seen[0][1].startswith("https://dashscope.aliyuncs.com/api/v1/uploads")


def test_presets_valid():
    for k in ("openai", "groq", "azure", "dashscope", "vllm-qwen3", "glm"):
        m = from_preset(k)
        assert m.type and (m.base_url or m.type == "dashscope_qwen")
    az = create_api_engine(from_preset("azure"), "k")
    assert "/openai/deployments/whisper/audio/transcriptions?api-version=" in az.url()
    assert az.auth_headers() == {"api-key": "k"}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg")
def test_api_engine_through_pipeline_with_local_diarization(tmp_path):
    import soundfile as sf

    from asr_dashboard.config.schema import AppConfig
    from asr_dashboard.core.pipeline import Pipeline, TaskRequest
    from tests.test_pipeline import FakeDiarizer, FakeEmbedder

    src = tmp_path / "a.wav"
    sf.write(src, chunk(40).samples, 16000)

    def h(req):
        return httpx.Response(200, json={"text": "x", "segments": [
            {"start": s, "end": s + 9.5, "text": f"sentence {s}."} for s in range(0, 40, 10)]})

    e, _ = engine(h)
    cfg = AppConfig()
    cfg.app.output_dir = str(tmp_path / "out")
    cfg.diarization.enabled = True
    from asr_dashboard.core.voiceprints.store import VoiceprintStore

    store = VoiceprintStore(tmp_path / "vp")
    p = store.add_person("赵六")
    store.add_sample(p.id, np.zeros(16000, np.float32), embedding=np.array([1.0, 0.0]), model_id="fake-emb")
    r = Pipeline(cfg, e, diarizer_provider=lambda c: FakeDiarizer(), voiceprint_store=store,
                 embedder_provider=lambda c: FakeEmbedder(), use_silero=False).run(TaskRequest(str(src))).result
    assert len(r.segments) == 4 and r.segments[1].start == 10
    assert [s.speaker for s in r.segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]
    assert any(i.display_name == "赵六" for i in r.speakers.values())

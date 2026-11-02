"""LLM post-processing with guard rails: chunking, JSON in/out, result validation, per-chunk rule fallback."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from asr_dashboard.models import CancelledError, CancelToken, Segment, TranscriptionResult
from asr_dashboard.text.llm_client import LLMClient, LLMError
from asr_dashboard.text.rules import RuleOptions, apply_rules

log = logging.getLogger(__name__)

CLEANUP_SYSTEM = """你是语音识别转录稿的校对员。输入是 JSON：items 为待校对条目，previous 为前文（只读，仅供理解语境）。
只修改每个条目的 text 字段，要求：
1. 修正同音错字、错别字和专有名词（优先使用 context 中给出的词汇和背景）；
2. 补全或修正标点；
3. 去掉口头禅（嗯、啊、呃、um、uh 等）和无意义的重复；
4. 不得增加或删除实际内容，不得改写语气或意译，不得翻译；
5. 不得合并、拆分、增删或重排条目，id 必须与输入完全一致。
只输出 JSON：{"items": [{"id": <id>, "text": "<修改后的文字>"}]}，不要输出任何其他内容。"""

SUMMARY_SYSTEM = """你是会议/访谈记录的整理助手。根据转录内容写一份中文 Markdown 摘要：
先用 2–4 句概括主题，再列出“## 要点”（项目符号），如有明确的结论、决定或待办，列出“## 结论与待办”。
只依据原文，不要编造。"""

SPEAKER_SYSTEM = """根据对话内容（称呼、自我介绍、他人的点名等）推测每个说话人编号可能对应的真实姓名或身份。
没有足够依据的不要猜。只输出 JSON：{"suggestions": [{"speaker": "SPEAKER_00", "name": "张三", "reason": "依据"}]}"""


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def change_ratio(old: str, new: str) -> float:
    return edit_distance(old, new) / max(1, len(old))


def make_chunks(segments: list[Segment], max_chars: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    cur: list[int] = []
    size = 0
    for i, s in enumerate(segments):
        n = len(s.text) + 30
        if cur and size + n > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    if cur:
        chunks.append(cur)
    return chunks


@dataclass
class ChunkOutcome:
    indices: list[int]
    texts: dict[int, str] | None  # None -> fall back to rules
    reason: str = ""


def check_response(data, ids: list[int], originals: dict[int, str], max_ratio: float) -> tuple[dict[int, str] | None, str]:
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return None, "返回结构不是条目数组"
    out: dict[int, str] = {}
    for it in items:
        if not isinstance(it, dict) or "id" not in it or not isinstance(it.get("text"), str):
            return None, "条目缺少 id 或 text"
        try:
            out[int(it["id"])] = it["text"].strip()
        except (TypeError, ValueError):
            return None, f"非法 id {it.get('id')!r}"
    if sorted(out) != sorted(ids) or len(items) != len(ids):
        return None, "返回的条目 id 与输入不一致"
    for i, new in out.items():
        r = change_ratio(originals[i], new)
        if r > max_ratio:
            return None, f"条目 {i} 改动比例 {r:.0%} 超过上限 {max_ratio:.0%}"
    return out, ""


def _speaker_label(result: TranscriptionResult, seg: Segment) -> str | None:
    return result.speaker_name(seg.speaker) if seg.speaker else None


def run_chunk(client: LLMClient, result: TranscriptionResult, idxs: list[int], context: str, max_ratio: float,
              cancel: CancelToken, temperature: float) -> ChunkOutcome:
    segs = result.segments
    prev_idx = [i for i in range(max(0, idxs[0] - 2), idxs[0])]
    payload = {
        "context": context or "",
        "previous": [{"id": i, "speaker": _speaker_label(result, segs[i]), "text": segs[i].text} for i in prev_idx],
        "items": [{"id": i, "speaker": _speaker_label(result, segs[i]), "text": segs[i].text} for i in idxs],
    }
    try:
        data = client.chat_json([{"role": "system", "content": CLEANUP_SYSTEM},
                                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                                cancel=cancel, temperature=temperature)
    except CancelledError:
        raise
    except LLMError as e:
        return ChunkOutcome(idxs, None, str(e))
    except Exception as e:  # noqa: BLE001
        return ChunkOutcome(idxs, None, f"{type(e).__name__}: {e}")
    texts, reason = check_response(data, idxs, {i: segs[i].text for i in idxs}, max_ratio)
    return ChunkOutcome(idxs, texts, reason)


def cleanup(result: TranscriptionResult, cfg, api_key: str, context: str, rule_opts: RuleOptions, progress,
            cancel: CancelToken, client: LLMClient | None = None) -> str:
    llm = cfg.llm
    client = client or LLMClient(llm.base_url, api_key, llm.model, llm.timeout, llm.json_mode, llm.temperature)
    full_ctx = context or ""  # task context + enabled glossaries (built by the pipeline)
    segs = result.segments
    chunks = make_chunks(segs, llm.chunk_chars)
    if not chunks:
        return "llm"
    outcomes: list[ChunkOutcome] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, llm.concurrency)) as ex:
        futs = [ex.submit(run_chunk, client, result, idxs, full_ctx, llm.max_change_ratio, cancel, llm.temperature)
                for idxs in chunks]
        for f in futs:
            outcomes.append(f.result())
            done += 1
            progress(done / len(chunks) * 0.85, f"大模型整理 {done}/{len(chunks)}")
            cancel.raise_if_cancelled()
    fell_back = 0
    for oc in outcomes:
        if oc.texts is None:
            fell_back += 1
            sub = [segs[i] for i in oc.indices]
            apply_rules(sub, rule_opts, merge=False)
            result.warnings.append(f"第 {oc.indices[0] + 1}–{oc.indices[-1] + 1} 条大模型整理失败（{oc.reason}），已改用规则处理")
            continue
        for i, new in oc.texts.items():
            if new != segs[i].text:
                if segs[i].original_text is None:
                    segs[i].original_text = segs[i].text
                segs[i].text = new
    result.segments = [s for s in segs if s.text.strip()]
    if fell_back == len(outcomes):
        return "rules"
    return "llm+rules" if fell_back else "llm"


def _transcript(result: TranscriptionResult) -> str:
    lines = []
    for s in result.segments:
        name = _speaker_label(result, s)
        lines.append(f"{name}: {s.text}" if name else s.text)
    return "\n".join(lines)


def summarize(result: TranscriptionResult, cfg, api_key: str, cancel: CancelToken, client: LLMClient | None = None) -> str:
    llm = cfg.llm
    client = client or LLMClient(llm.base_url, api_key, llm.model, llm.timeout, False, llm.temperature)
    text = _transcript(result)
    budget = max(2000, int(llm.context_chars * 0.7))
    if len(text) <= budget:
        return client.chat([{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": text}],
                           want_json=False, cancel=cancel).strip()
    parts = [text[i:i + budget] for i in range(0, len(text), budget)]
    partials = []
    for k, p in enumerate(parts, 1):
        cancel.raise_if_cancelled()
        partials.append(client.chat([{"role": "system", "content": SUMMARY_SYSTEM},
                                     {"role": "user", "content": f"（第 {k}/{len(parts)} 部分）\n{p}"}],
                                    want_json=False, cancel=cancel).strip())
    joined = "\n\n".join(partials)
    return client.chat([{"role": "system", "content": SUMMARY_SYSTEM + "\n输入是分块摘要，请合并成一份完整摘要，去掉重复。"},
                        {"role": "user", "content": joined[:budget]}], want_json=False, cancel=cancel).strip()


def guess_speakers(result: TranscriptionResult, cfg, api_key: str, cancel: CancelToken,
                   client: LLMClient | None = None) -> list[dict]:
    llm = cfg.llm
    client = client or LLMClient(llm.base_url, api_key, llm.model, llm.timeout, llm.json_mode, llm.temperature)
    open_labels = [k for k, v in result.speakers.items() if not v.voiceprint_id]
    if not open_labels:
        return []
    budget = max(2000, int(llm.context_chars * 0.7))
    data = client.chat_json([{"role": "system", "content": SPEAKER_SYSTEM},
                             {"role": "user", "content": f"需要推测的说话人：{', '.join(open_labels)}\n\n"
                                                         f"{_transcript(result)[:budget]}"}], cancel=cancel)
    out = []
    for s in (data.get("suggestions") if isinstance(data, dict) else data) or []:
        if isinstance(s, dict) and s.get("speaker") in open_labels and s.get("name"):
            out.append({"speaker": s["speaker"], "name": str(s["name"]), "reason": str(s.get("reason", ""))})
    return out

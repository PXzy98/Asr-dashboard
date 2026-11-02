"""Long-audio chunking (duration / vad / hybrid) and de-duplication of overlapping chunk results."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from asr_dashboard.core import vad as vadmod
from asr_dashboard.core.audio import SAMPLE_RATE, slice_samples
from asr_dashboard.models import AudioChunk, Segment, Word


@dataclass
class SegParams:
    strategy: str = "hybrid"
    target_duration: float = 60.0
    min_segment: float = 5.0
    max_segment: float = 60.0
    overlap: float = 0.5
    silence_threshold_db: float = -40.0
    min_silence: float = 0.3
    search_window: float = 10.0

    @classmethod
    def from_config(cls, seg, max_cap: float | None = None) -> SegParams:
        p = cls(
            strategy=seg.strategy, target_duration=seg.target_duration, min_segment=seg.min_segment,
            max_segment=seg.max_segment, overlap=seg.overlap, silence_threshold_db=seg.silence_threshold_db,
            min_silence=seg.min_silence, search_window=getattr(seg, "search_window", 10.0),
        )
        if max_cap:
            p.max_segment = min(p.max_segment, max_cap)
            p.target_duration = min(p.target_duration, p.max_segment)
        p.target_duration = min(max(p.target_duration, p.min_segment), p.max_segment)
        return p


@dataclass
class Bound:
    start: float
    end: float
    overlap_prev: float = 0.0  # seconds shared with the previous chunk (hard cut)


# ---------------------------------------------------------------------------------- strategies
def split_duration(total: float, p: SegParams) -> list[Bound]:
    dur = max(1.0, min(p.target_duration, p.max_segment))
    ov = min(max(0.0, p.overlap), dur / 4)
    if total <= dur:
        return [Bound(0.0, total)] if total > 0 else []
    out: list[Bound] = []
    start = 0.0
    while start < total:
        end = min(total, start + dur)
        out.append(Bound(start, end, ov if out else 0.0))
        if end >= total:
            break
        start = end - ov
    _merge_short_tail(out, p)
    return out


def split_hybrid(total: float, silences: list[tuple[float, float]], p: SegParams) -> list[Bound]:
    if total <= 0:
        return []
    if total <= p.max_segment:
        return [Bound(0.0, total)]
    out: list[Bound] = []
    cursor = 0.0
    pending_overlap = 0.0
    while total - cursor > p.max_segment:
        target = cursor + p.target_duration
        lo = max(cursor + p.min_segment, target - p.search_window)
        hi = min(cursor + p.max_segment, target + p.search_window)
        best = None
        best_len = 0.0
        for a, b in silences:
            # portion of the silence inside the window
            aa, bb = max(a, lo), min(b, hi)
            if bb - aa <= 0 or (b - a) < p.min_silence:
                continue
            length = bb - aa
            if length > best_len:
                best_len, best = length, (aa + bb) / 2
        if best is not None:
            out.append(Bound(cursor, best, pending_overlap))
            cursor, pending_overlap = best, 0.0
        else:  # hard cut at max length, overlap with next chunk
            cut = cursor + p.max_segment
            out.append(Bound(cursor, cut, pending_overlap))
            ov = min(p.overlap, p.max_segment / 4)
            cursor, pending_overlap = cut - ov, ov
    out.append(Bound(cursor, total, pending_overlap))
    _merge_short_tail(out, p)
    return out


def split_vad(total: float, regions: list[tuple[float, float]], p: SegParams, pad: float = 0.2) -> list[Bound]:
    """Group speech regions into chunks of ~target length; silence between chunks is skipped."""
    if total <= 0 or not regions:
        return []
    regions = [(max(0.0, a - pad), min(total, b + pad)) for a, b in regions]
    # split overly long regions (hard cuts with overlap)
    pieces: list[Bound] = []
    for a, b in regions:
        if b - a <= p.max_segment:
            pieces.append(Bound(a, b))
        else:
            for bd in split_duration(b - a, SegParams(target_duration=p.max_segment, max_segment=p.max_segment,
                                                      min_segment=p.min_segment, overlap=p.overlap)):
                pieces.append(Bound(a + bd.start, a + bd.end, bd.overlap_prev))
    out: list[Bound] = []
    for pc in pieces:
        if out and pc.overlap_prev == 0.0 and pc.end - out[-1].start <= p.target_duration:
            out[-1] = Bound(out[-1].start, max(out[-1].end, pc.end), out[-1].overlap_prev)
            continue
        if out and pc.overlap_prev == 0.0 and pc.start < out[-1].end:
            pc = Bound(out[-1].end, pc.end)  # padding made them touch: no double-coverage
            if pc.end - pc.start <= 0:
                continue
        out.append(pc)
    return out


def _merge_short_tail(out: list[Bound], p: SegParams) -> None:
    if len(out) >= 2 and (out[-1].end - out[-1].start) < p.min_segment:
        last = out.pop()
        prev = out[-1]
        if last.end - prev.start <= p.max_segment * 1.5:
            out[-1] = Bound(prev.start, last.end, prev.overlap_prev)
        else:
            out.append(last)


# ---------------------------------------------------------------------------------- facade
def plan_chunks(samples: np.ndarray, p: SegParams, sr: int = SAMPLE_RATE,
                use_silero: bool = True) -> list[Bound]:
    total = len(samples) / sr
    if p.strategy == "duration":
        return split_duration(total, p)
    if p.strategy == "vad":
        regions = vadmod.speech_regions(samples, sr, min_silence=p.min_silence, use_silero=use_silero)
        bounds = split_vad(total, regions, p)
        return bounds or ([Bound(0.0, total)] if total <= p.max_segment else split_duration(total, p))
    # hybrid
    if total <= p.max_segment:
        return [Bound(0.0, total)]
    sil = vadmod.silences(samples, sr, threshold_db=p.silence_threshold_db, min_silence=p.min_silence)
    if use_silero and vadmod.silero_available():
        try:
            regions = vadmod.speech_regions(samples, sr, min_silence=p.min_silence, use_silero=True)
            gaps = [(a[1], b[0]) for a, b in zip(regions, regions[1:], strict=False) if b[0] - a[1] >= p.min_silence]
            sil = sorted(set(sil) | set(gaps))
        except Exception:  # noqa: BLE001
            pass
    return split_hybrid(total, sil, p)


def make_chunks(samples: np.ndarray, bounds: list[Bound], sr: int = SAMPLE_RATE) -> list[AudioChunk]:
    chunks = []
    for i, b in enumerate(bounds):
        seg = slice_samples(samples, b.start, b.end, sr)
        chunks.append(AudioChunk(offset=b.start, duration=len(seg) / sr, samples=seg, sample_rate=sr, index=i))
    return chunks


# ---------------------------------------------------------------------------------- dedup
_TOKEN_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]|[A-Za-z0-9']+|[^\sA-Za-z0-9']")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _norm(tok: str) -> str:
    return tok.lower().strip(".,!?;:，。！？；：、\"'")


def text_overlap_len(prev: str, nxt: str, max_tokens: int = 40, min_tokens: int = 2) -> int:
    """Number of leading tokens of ``nxt`` that repeat the trailing tokens of ``prev``."""
    a = [t for t in (_norm(x) for x in _tokens(prev)) if t]
    b_raw = _tokens(nxt)
    b = [_norm(x) for x in b_raw]
    b_idx = [i for i, t in enumerate(b) if t]
    b = [b[i] for i in b_idx]
    best = 0
    for k in range(min(max_tokens, len(a), len(b)), min_tokens - 1, -1):
        if a[-k:] == b[:k]:
            best = k
            break
    if best == 0:
        return 0
    # translate back to raw-token count (including punctuation tokens skipped)
    return b_idx[best - 1] + 1


def strip_leading_tokens(text: str, n_tokens: int) -> str:
    if n_tokens <= 0:
        return text
    for count, m in enumerate(_TOKEN_RE.finditer(text), start=1):
        if count == n_tokens:
            return text[m.end():].lstrip(" ，,。.、")
    return ""


def dedup_chunk_segments(per_chunk: list[list[Segment]], bounds: list[Bound]) -> list[Segment]:
    """Remove text duplicated in the overlap between consecutive hard-cut chunks."""
    out: list[Segment] = []
    for idx, segs in enumerate(per_chunk):
        segs = [s for s in segs if s.text.strip() or s.words]
        if idx > 0 and bounds[idx].overlap_prev > 0 and out:
            ov_start = bounds[idx].start
            ov_end = bounds[idx].start + bounds[idx].overlap_prev
            middle = (ov_start + ov_end) / 2
            has_words = any(s.words for s in segs) and any(s.words for s in out)
            if has_words:
                # previous chunk keeps words before the middle of the overlap, this one after it
                for s in out[::-1]:
                    if s.end <= ov_start - 1e-6:
                        break
                    _trim_words(s, keep=lambda w, m=middle: (w.start + w.end) / 2 < m)
                for s in segs:
                    if s.start >= ov_end:
                        break
                    _trim_words(s, keep=lambda w, m=middle: (w.start + w.end) / 2 >= m)
                out = [s for s in out if s.text.strip()]
                segs = [s for s in segs if s.text.strip()]
            elif segs:
                n = text_overlap_len(out[-1].text, segs[0].text)
                if n:
                    segs[0].text = strip_leading_tokens(segs[0].text, n)
                    if not segs[0].text.strip():
                        segs = segs[1:]
        out.extend(segs)
    return out


def _join_words(words: list[Word]) -> str:
    text = ""
    for w in words:
        t = w.text
        if text and _needs_space(text[-1], t[:1]):
            text += " "
        text += t
    return text


def _needs_space(a: str, b: str) -> bool:
    if not a or not b:
        return False
    cjk = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
    if cjk.match(a) or cjk.match(b):
        return False
    return not (b in ".,!?;:%)]}" or a in "([{")


def _trim_words(seg: Segment, keep) -> None:
    if not seg.words:
        return
    kept = [w for w in seg.words if keep(w)]
    if len(kept) == len(seg.words):
        return
    seg.words = kept
    if kept:
        seg.start, seg.end = kept[0].start, kept[-1].end
        seg.text = _join_words(kept)
    else:
        seg.text = ""


join_words = _join_words

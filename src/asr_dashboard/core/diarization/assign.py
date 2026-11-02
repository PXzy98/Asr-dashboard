"""Turn post-processing and speaker <-> text alignment (word level or segment level)."""

from __future__ import annotations

from asr_dashboard.core.diarization.base import SpeakerTurn
from asr_dashboard.core.segmentation import join_words
from asr_dashboard.models import Segment


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def clean_turns(turns: list[SpeakerTurn], min_turn: float = 0.3, merge_gap: float = 0.5) -> list[SpeakerTurn]:
    """Merge same-speaker turns separated by < merge_gap; absorb turns shorter than min_turn into neighbours."""
    ts = sorted((SpeakerTurn(t.start, t.end, t.speaker) for t in turns if t.end > t.start), key=lambda t: t.start)
    if not ts:
        return []

    def merge(seq: list[SpeakerTurn]) -> list[SpeakerTurn]:
        out: list[SpeakerTurn] = []
        for t in seq:
            if out and out[-1].speaker == t.speaker and t.start - out[-1].end < merge_gap:
                out[-1].end = max(out[-1].end, t.end)
            else:
                out.append(SpeakerTurn(t.start, t.end, t.speaker))
        return out

    ts = merge(ts)
    if min_turn > 0 and len(ts) > 1:
        for i, t in enumerate(ts):
            if t.duration < min_turn:
                prev = ts[i - 1] if i > 0 else None
                nxt = ts[i + 1] if i + 1 < len(ts) else None
                cands = [c for c in (prev, nxt) if c is not None and c.duration >= min_turn]
                if not cands:
                    cands = [c for c in (prev, nxt) if c is not None]
                # choose the closer neighbour (ties -> previous)
                best = min(cands, key=lambda c: abs((c.start + c.end) / 2 - (t.start + t.end) / 2))
                t.speaker = best.speaker
        ts = merge(ts)
    return ts


def overlaps_from_turns(turns: list[SpeakerTurn], min_dur: float = 0.2) -> list[tuple[float, float, list[str]]]:
    """Regions where two or more speakers talk at once (sweep line)."""
    events = []
    for t in turns:
        events.append((t.start, 1, t.speaker))
        events.append((t.end, -1, t.speaker))
    events.sort(key=lambda e: (e[0], e[1]))
    active: dict[str, int] = {}
    out: list[tuple[float, float, list[str]]] = []
    last_t = None
    for time, kind, spk in events:
        live = sorted(s for s, c in active.items() if c > 0)
        if last_t is not None and len(live) >= 2 and time - last_t > 0:
            if out and abs(out[-1][1] - last_t) < 1e-6 and out[-1][2] == live:
                out[-1] = (out[-1][0], time, live)
            else:
                out.append((last_t, time, live))
        active[spk] = active.get(spk, 0) + kind
        last_t = time
    return [(a, b, s) for a, b, s in out if b - a >= min_dur]


def _speaker_for(t0: float, t1: float, turns: list[SpeakerTurn]) -> str | None:
    best, best_ov = None, 0.0
    for t in turns:
        if t.start > t1:
            break
        ov = _overlap(t0, t1, t.start, t.end)
        if ov > best_ov:
            best, best_ov = t.speaker, ov
    if best is not None:
        return best
    # no overlap: nearest turn within 1 s
    mid = (t0 + t1) / 2
    near = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end)), default=None)
    if near and min(abs(mid - near.start), abs(mid - near.end)) <= 1.0:
        return near.speaker
    return None


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn],
                    multi_threshold: float = 0.3) -> tuple[list[Segment], list[str]]:
    """Attach speakers to segments. Word-level when words exist (splitting at speaker changes)."""
    turns = sorted(turns, key=lambda t: t.start)
    warnings: list[str] = []
    if not turns:
        return segments, warnings
    out: list[Segment] = []
    ambiguous = 0
    for seg in segments:
        if seg.words:
            for w in seg.words:
                w.speaker = _speaker_for(w.start, w.end, turns)
            # fill unknown words with neighbour speaker
            last = None
            for w in seg.words:
                if w.speaker is None:
                    w.speaker = last
                last = w.speaker or last
            first_known = next((w.speaker for w in seg.words if w.speaker), None)
            for w in seg.words:
                if w.speaker is None:
                    w.speaker = first_known
            # split on change
            cur: list = []
            for w in seg.words:
                if cur and w.speaker != cur[-1].speaker:
                    out.append(_from_words(seg, cur))
                    cur = []
                cur.append(w)
            if cur:
                out.append(_from_words(seg, cur))
        else:
            shares: dict[str, float] = {}
            for t in turns:
                if t.start > seg.end:
                    break
                ov = _overlap(seg.start, seg.end, t.start, t.end)
                if ov > 0:
                    shares[t.speaker] = shares.get(t.speaker, 0.0) + ov
            if shares:
                seg.speaker = max(shares, key=shares.get)
                dur = max(seg.duration, 1e-6)
                if sum(1 for v in shares.values() if v / dur > multi_threshold) >= 2:
                    ambiguous += 1
            else:
                seg.speaker = _speaker_for(seg.start, seg.end, turns)
            out.append(seg)
    if ambiguous:
        warnings.append(
            f"有 {ambiguous} 个分段同时包含两位以上说话人（各占 30% 以上），整段只能归给一人。"
            "建议开启强制对齐以获得更准确的说话人切分"
        )
    return out, warnings


def _from_words(seg: Segment, words: list) -> Segment:
    return Segment(text=join_words(words), start=words[0].start, end=words[-1].end, speaker=words[0].speaker,
                   language=seg.language, words=list(words))


def speaker_clean_regions(speaker: str, segments: list[Segment], overlaps, target: float = 30.0,
                          max_regions: int = 8) -> list[tuple[float, float]]:
    """Pick the longest stretches of ``speaker`` that don't overlap other speakers (for voiceprint samples)."""
    regions = []
    for s in segments:
        if s.speaker != speaker or s.duration < 1.0:
            continue
        pieces = [(s.start, s.end)]
        for a, b, _ in overlaps:  # cut out multi-speaker stretches
            nxt = []
            for x, y in pieces:
                if b <= x or a >= y:
                    nxt.append((x, y))
                    continue
                if a > x:
                    nxt.append((x, a))
                if b < y:
                    nxt.append((b, y))
            pieces = nxt
        regions.extend((x, y) for x, y in pieces if y - x >= 1.0)
    regions.sort(key=lambda r: r[1] - r[0], reverse=True)
    picked, total = [], 0.0
    for r in regions:
        if total >= target or len(picked) >= max_regions:
            break
        picked.append(r)
        total += r[1] - r[0]
    return sorted(picked)

"""Sentence-level re-segmentation from word timestamps + punctuation re-attachment."""

from __future__ import annotations

import re

from asr_dashboard.core.segmentation import join_words
from asr_dashboard.models import Segment, Word

SENT_END = set("。！？!?.…")
_PUNCT = set("，。！？；：、,.!?;:…\"'”’）)》」』】-—~～")
_WORDCHAR = re.compile(r"[\w぀-ヿ㐀-鿿가-힯]")


def attach_punctuation(text: str, items: list[tuple[str, float, float]]) -> list[Word]:
    """Map aligner tokens (which carry no punctuation) back onto the recognised text.

    Punctuation following a token in ``text`` is appended to that token so that sentence splitting
    and export keep the original punctuation.
    """
    words: list[Word] = []
    lower = text.lower()
    cursor = 0
    for tok, st, en in items:
        t = tok.strip()
        if not t:
            continue
        idx = lower.find(t.lower(), cursor)
        if idx < 0 or idx - cursor > 50:  # not found nearby: keep raw token
            words.append(Word(t, float(st), float(en)))
            continue
        end = idx + len(t)
        # punctuation directly preceding (e.g. opening quotes) is ignored; trailing punctuation kept
        j = end
        while j < len(text) and (text[j] in _PUNCT or (text[j].isspace() and j + 1 < len(text) and text[j + 1] in _PUNCT)):
            j += 1
        trailing = text[end:j].strip()
        words.append(Word(text[idx:end] + trailing, float(st), float(en)))
        cursor = j
    return words


def split_sentences(words: list[Word], max_duration: float = 15.0, pause: float = 0.6,
                    language: str | None = None) -> list[Segment]:
    """Split a word stream on sentence-final punctuation, pauses > ``pause`` s or length > ``max_duration``."""
    segs: list[Segment] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            segs.append(Segment(text=join_words(cur), start=cur[0].start, end=cur[-1].end,
                                language=language, words=list(cur)))
            cur.clear()

    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            too_long = (w.end - cur[0].start) > max_duration
            if gap > pause or too_long:
                flush()
        cur.append(w)
        if w.text and w.text.rstrip("\"'”’）)」』")[-1:] in SENT_END:
            flush()
    flush()
    return segs

"""Rule-based post-processing (used when no LLM is configured, or as per-chunk fallback)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from asr_dashboard.models import Segment

CJK = r"぀-ヿ㐀-鿿가-힯"
_CJK_RE = re.compile(f"[{CJK}]")
ZH_END = "。！？；…"
EN_END = ".!?;"
_HALF2FULL = {",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："}
_FULL2HALF = {v: k for k, v in _HALF2FULL.items()} | {"、": ",", "（": "(", "）": ")", "“": '"', "”": '"'}


@dataclass
class RuleOptions:
    merge_sentences: bool = True
    normalize_punctuation: bool = True
    clean_whitespace: bool = True
    remove_fillers: bool = False
    smart_capitalize: bool = True
    filler_words: dict[str, list[str]] = field(default_factory=dict)
    merge_max_gap: float = 1.0
    merge_max_duration: float = 30.0

    @classmethod
    def from_config(cls, t) -> RuleOptions:
        return cls(t.merge_sentences, t.normalize_punctuation, getattr(t, "clean_whitespace", True), t.remove_fillers,
                   t.smart_capitalize, dict(t.filler_words))


def is_cjk_text(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if _CJK_RE.match(c)) / len(letters) >= 0.3


def lang_of(seg: Segment) -> str:
    if seg.language in ("zh", "yue", "ja", "ko"):
        return seg.language
    if seg.language:
        return seg.language if not is_cjk_text(seg.text) else "zh"
    return "zh" if is_cjk_text(seg.text) else "en"


# -------------------------------------------------------------------------------- primitives
def normalize_punctuation(text: str, lang: str) -> str:
    if lang in ("zh", "yue", "ja"):
        # half-width punctuation next to CJK -> full-width (keep decimals like 3.5 and times 10:30)
        def repl(m: re.Match) -> str:
            return m.group(1) + _HALF2FULL[m.group(2)]

        text = re.sub(f"([{CJK}])\\s*([,.?!;:])(?!\\d)", repl, text)
        text = re.sub(f"(?<![\\d])([,?!;])\\s*(?=[{CJK}])", lambda m: _HALF2FULL[m.group(1)], text)
        text = re.sub(r"([，。！？；：])[,.?!;:]+", r"\1", text)
    else:
        for full, half in _FULL2HALF.items():
            text = text.replace(full, half)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([,!?;:])(?=[A-Za-z])", r"\1 ", text)
    # collapse repeated punctuation (keep "..." / "……" ellipses)
    text = re.sub(r"([，。！？；：、,!?;:])\1+", r"\1", text)
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)
    text = re.sub(r"[，,]\s*([。！？.!?])", r"\1", text)
    return text


def clean_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(f"(?<=[{CJK}，。！？；：、])\\s+(?=[{CJK}，。！？；：、])", "", text)
    text = re.sub(f"(?<=[{CJK}])\\s+(?=[，。！？；：、])", "", text)
    text = re.sub(f"([{CJK}])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(f"([A-Za-z0-9%])([{CJK}])", r"\1 \2", text)
    return text


def capitalize_sentences(text: str) -> str:
    def up(m: re.Match) -> str:
        return m.group(1) + m.group(2).upper()

    text = re.sub(r"^(\s*[\"'(]?)([a-z])", up, text)
    text = re.sub(r"([.!?]\s+[\"'(]?)([a-z])", up, text)
    text = re.sub(r"\bi\b(?=[\s',.!?])", "I", text)
    return text


def remove_fillers(text: str, lang: str, words: dict[str, list[str]]) -> str:
    if lang in ("zh", "yue"):
        for w in sorted(words.get("zh", []), key=len, reverse=True):
            if len(w) == 1:  # interjections: remove with following punctuation anywhere
                text = re.sub(f"{re.escape(w)}+[，,、。]?", "", text)
            else:  # word fillers only when used as a pause ("那个，", "就是，") or repeated
                text = re.sub(f"(?:(?<=^)|(?<=[，,。！？\\s])){re.escape(w)}[，,、]", "", text)
                text = re.sub(f"({re.escape(w)})(?:[，,]?\\1)+", r"\1", text)
        text = re.sub(r"^[，,、\s]+", "", text)
    elif lang == "ja":
        for w in words.get("ja", []):
            text = re.sub(f"{re.escape(w)}[、,]?\\s*", "", text)
    else:
        for w in sorted(words.get("en", []), key=len, reverse=True):
            text = re.sub(rf"(?i)(^|[\s,]){re.escape(w)}\b[,]?", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"^\s*,\s*", "", text)
        text = re.sub(r"\s+,", ",", text)
    return text.strip()


def ends_sentence(text: str) -> bool:
    t = text.rstrip("\"'”’）)」』 ")
    return bool(t) and (t[-1] in ZH_END or t[-1] in EN_END)


def merge_segments(segs: list[Segment], max_gap: float = 1.0, max_duration: float = 30.0) -> list[Segment]:
    """Merge short fragments into whole sentences; never across speakers."""
    out: list[Segment] = []
    for s in segs:
        if (out and out[-1].speaker == s.speaker and not ends_sentence(out[-1].text)
                and s.start - out[-1].end <= max_gap and s.end - out[-1].start <= max_duration):
            prev = out[-1]
            sep = "" if (is_cjk_text(prev.text) or is_cjk_text(s.text)) else " "
            prev.text = (prev.text.rstrip() + sep + s.text.lstrip()).strip()
            if prev.original_text is not None or s.original_text is not None:
                prev.original_text = ((prev.original_text or prev.text) + sep + (s.original_text or s.text)).strip()
            prev.end = max(prev.end, s.end)
            prev.words = prev.words + s.words
        else:
            out.append(Segment(s.text, s.start, s.end, s.speaker, s.language, list(s.words), s.original_text))
    return out


def clean_text(text: str, lang: str, opts: RuleOptions) -> str:
    if opts.remove_fillers:
        text = remove_fillers(text, lang, opts.filler_words)
    if opts.normalize_punctuation:
        text = normalize_punctuation(text, lang)
    if opts.clean_whitespace:
        text = clean_whitespace(text)
    if opts.smart_capitalize and lang not in ("zh", "yue", "ja", "ko"):
        text = capitalize_sentences(text)
    return text


def apply_rules(segs: list[Segment], opts: RuleOptions, merge: bool | None = None) -> list[Segment]:
    """Clean each segment's text; optionally merge fragments. Keeps original_text for undo."""
    out = []
    for s in segs:
        new = clean_text(s.text, lang_of(s), opts)
        if new != s.text and s.original_text is None:
            s.original_text = s.text
        s.text = new
        if s.text.strip():
            out.append(s)
    if opts.merge_sentences if merge is None else merge:
        out = merge_segments(out, opts.merge_max_gap, opts.merge_max_duration)
    return out

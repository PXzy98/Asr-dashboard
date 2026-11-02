"""Context prompt (hot words) assembly: background text first, then a de-duplicated term list."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextResult:
    text: str
    truncated: bool
    warnings: list[str]


def _norm_lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def build_context(default_context: str = "", glossaries: list | None = None, task_context: str = "",
                  max_chars: int = 2000) -> ContextResult:
    """Merge global default context, selected glossaries and per-task context.

    Background descriptions (default, glossary descriptions, task text) come first, followed by
    ``术语：a、b、c`` with duplicates removed (case-insensitive, order preserved).
    """
    backgrounds: list[str] = []
    terms: list[str] = []
    seen_bg: set[str] = set()
    seen_terms: set[str] = set()

    def add_bg(t: str) -> None:
        for ln in _norm_lines(t):
            if ln.lower() not in seen_bg:
                seen_bg.add(ln.lower())
                backgrounds.append(ln)

    def add_term(t: str) -> None:
        t = t.strip().strip("，,、;；")
        if t and t.lower() not in seen_terms:
            seen_terms.add(t.lower())
            terms.append(t)

    add_bg(default_context)
    for g in glossaries or []:
        desc = getattr(g, "description", "") if not isinstance(g, dict) else g.get("description", "")
        gterms = getattr(g, "terms", None) if not isinstance(g, dict) else g.get("terms")
        add_bg(desc)
        for t in gterms or []:
            add_term(t)
    add_bg(task_context)

    parts = []
    if backgrounds:
        parts.append("\n".join(backgrounds))
    if terms:
        parts.append("术语：" + "、".join(terms))
    text = "\n".join(parts)
    warnings: list[str] = []
    truncated = False
    if len(text) > max_chars:
        truncated = True
        cut = text[:max_chars]
        # don't leave half a term behind
        sep = max(cut.rfind("、"), cut.rfind("\n"))
        if sep > max_chars * 0.6:
            cut = cut[:sep]
        warnings.append(f"上下文提示共 {len(text)} 字，超过上限 {max_chars} 字，已截断（{len(cut)} 字）")
        text = cut
    return ContextResult(text=text, truncated=truncated, warnings=warnings)

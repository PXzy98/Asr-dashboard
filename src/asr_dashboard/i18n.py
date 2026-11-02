"""Minimal i18n: UI strings are written in Chinese and looked up through ``tr``.

To add English later, fill ``_TRANSLATIONS["en"]`` with ``{"中文原文": "English"}`` and call
``set_language("en")`` at startup. Untranslated strings fall back to the source text.
"""

from __future__ import annotations

_current = "zh"
_TRANSLATIONS: dict[str, dict[str, str]] = {"zh": {}, "en": {}}


def set_language(lang: str) -> None:
    global _current
    _current = lang if lang in _TRANSLATIONS else "zh"


def tr(text: str, **kwargs) -> str:
    s = _TRANSLATIONS.get(_current, {}).get(text, text)
    return s.format(**kwargs) if kwargs else s

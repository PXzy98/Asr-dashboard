"""Secret masking for logs, error dialogs and exported diagnostics."""

from __future__ import annotations

import logging
import re
import threading

_lock = threading.Lock()
_known_secrets: set[str] = set()

# Generic patterns: bearer tokens, api-key headers, common key prefixes, key=... query params
_PATTERNS = [
    re.compile(r"(?i)(authorization\s*[:=]\s*['\"]?bearer\s+)([^\s'\",}]+)"),
    re.compile(r"(?i)((?:api[-_]?key|x-api-key|token|access_token|hf_token)['\"]?\s*[:=]\s*['\"]?)([^\s'\",}&]+)"),
    re.compile(r"\b(sk-[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-]{6,}"),
    re.compile(r"\b(hf_[A-Za-z0-9]{2})[A-Za-z0-9]{10,}"),
    re.compile(r"\b(gsk_[A-Za-z0-9]{2})[A-Za-z0-9]{10,}"),
]


def mask(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 8:
        return "****"
    return f"{secret[:3]}****{secret[-2:]}"


def register_secret(secret: str | None) -> None:
    """Remember a concrete secret value so it is always replaced, whatever context it appears in."""
    if secret and len(secret) >= 4:
        with _lock:
            _known_secrets.add(secret)


def redact(text: str) -> str:
    if not text:
        return text
    with _lock:
        secrets = sorted(_known_secrets, key=len, reverse=True)
    for s in secrets:
        if s in text:
            text = text.replace(s, mask(s))
    for i, pat in enumerate(_PATTERNS):
        if i < 2:
            text = pat.sub(lambda m: m.group(1) + ("****" if "****" not in m.group(2) else m.group(2)), text)
        else:
            text = pat.sub(lambda m: m.group(1) + "****", text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        red = redact(msg)
        if red != msg:
            record.msg = red
            record.args = ()
        if record.exc_info and record.exc_info[1] is not None:
            # exception text is rendered later; pre-render and redact it
            import traceback

            record.exc_text = redact("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        return True

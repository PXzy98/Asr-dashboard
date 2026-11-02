"""Single entry point for text post-processing: LLM when a validated LLM is configured, else rules."""

from __future__ import annotations

import logging
from collections.abc import Callable

from asr_dashboard.models import CancelToken, TranscriptionResult
from asr_dashboard.text.rules import RuleOptions, apply_rules

log = logging.getLogger(__name__)


def select_mode(cfg) -> str:
    """'llm' only when the LLM settings were saved+validated and the switch is on."""
    return "llm" if cfg.llm_active and cfg.llm.tasks.cleanup else "rules"


def mode_label(result: TranscriptionResult, cfg) -> str:
    if result.postprocess_mode.startswith("llm"):
        suffix = "（部分回退规则）" if result.postprocess_mode == "llm+rules" else ""
        return f"大模型：{cfg.llm.model}{suffix}"
    return "规则"


def run_postprocess(result: TranscriptionResult, cfg, api_key: str = "", context: str = "",
                    progress: Callable[[float, str], None] | None = None, cancel: CancelToken | None = None,
                    client=None) -> TranscriptionResult:
    progress = progress or (lambda *_: None)
    cancel = cancel or CancelToken()
    opts = RuleOptions.from_config(cfg.text)
    mode = select_mode(cfg)
    if mode == "llm":
        from asr_dashboard.text import llm_postprocess

        try:
            mode = llm_postprocess.cleanup(result, cfg, api_key, context, opts, progress, cancel, client=client)
        except Exception as e:  # noqa: BLE001 - any failure falls back to rules for everything
            if type(e).__name__ == "CancelledError":
                raise
            log.exception("大模型后处理失败")
            result.warnings.append(f"大模型后处理失败，已全部改用规则处理：{e}")
            result.segments = apply_rules(result.segments, opts)
            mode = "rules"
    else:
        result.segments = apply_rules(result.segments, opts)
    result.postprocess_mode = mode
    if cfg.llm_active and (cfg.llm.tasks.summary or cfg.llm.tasks.speaker_guess):
        from asr_dashboard.text import llm_postprocess

        if cfg.llm.tasks.summary:
            try:
                progress(0.9, "生成摘要")
                result.summary = llm_postprocess.summarize(result, cfg, api_key, cancel, client=client)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "CancelledError":
                    raise
                result.warnings.append(f"摘要生成失败：{e}")
        if cfg.llm.tasks.speaker_guess:
            try:
                progress(0.95, "推测说话人身份")
                result.speaker_suggestions = llm_postprocess.guess_speakers(result, cfg, api_key, cancel, client=client)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "CancelledError":
                    raise
                result.warnings.append(f"说话人身份推测失败：{e}")
    result.rebuild_text()
    return result


def undo_segment(result: TranscriptionResult, index: int) -> None:
    s = result.segments[index]
    if s.original_text is not None:
        s.text = s.original_text
        s.original_text = None
    result.rebuild_text()


def undo_all(result: TranscriptionResult) -> int:
    n = 0
    for i, s in enumerate(result.segments):
        if s.original_text is not None:
            undo_segment(result, i)
            n += 1
    return n

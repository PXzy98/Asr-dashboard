"""Shared definitions for the local Qwen3-ASR engine and its backends."""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from asr_dashboard.config.schema import ALIGNER_LANGUAGES, QWEN_LANGUAGES

log = logging.getLogger(__name__)

LANG_CODE_TO_NAME = dict(QWEN_LANGUAGES)
LANG_NAME_TO_CODE = {v.lower(): k for k, v in QWEN_LANGUAGES.items()}

# The qwen-asr package itself cuts timestamp inputs at 180 s (MAX_FORCE_ALIGN_INPUT_SECONDS) although the
# model card says "up to 5 minutes". We cap aligned chunks at 180 s, which also satisfies the <=300 s rule.
ALIGNER_MAX_SECONDS = 180.0
ASR_MAX_SECONDS = 1200.0  # qwen-asr MAX_ASR_INPUT_SECONDS
MIN_TRANSFORMERS_NATIVE = (5, 13)

DIALECTS = ("安徽、东北、福建、甘肃、贵州、河北、河南、湖北、湖南、江西、宁夏、山东、陕西、山西、四川、天津、"
            "云南、浙江、粤语（港式）、粤语（广式）、吴语、闽南语")

REPO_IDS = {
    ("1.7b", False): "Qwen/Qwen3-ASR-1.7B",
    ("0.6b", False): "Qwen/Qwen3-ASR-0.6B",
    ("1.7b", True): "Qwen/Qwen3-ASR-1.7B-hf",
    ("0.6b", True): "Qwen/Qwen3-ASR-0.6B-hf",
}
ALIGNER_REPO = {False: "Qwen/Qwen3-ForcedAligner-0.6B", True: "Qwen/Qwen3-ForcedAligner-0.6B-hf"}


def lang_name(code_or_name: str | None) -> str | None:
    if not code_or_name or code_or_name == "auto":
        return None
    if code_or_name in LANG_CODE_TO_NAME:
        return LANG_CODE_TO_NAME[code_or_name]
    n = code_or_name.strip()
    return n[:1].upper() + n[1:].lower()


def lang_code(name: str | None) -> str | None:
    if not name:
        return None
    return LANG_NAME_TO_CODE.get(name.lower(), name)


def aligner_supports(language_name: str | None) -> bool:
    return bool(language_name) and language_name in ALIGNER_LANGUAGES


@dataclass
class LoadParams:
    model_path: str
    aligner_path: str | None
    device: str
    dtype: str  # resolved name
    attn_implementation: str | None
    aligner_device: str
    aligner_dtype: str
    aligner_attn: str | None
    max_inference_batch_size: int
    max_new_tokens: int
    gpu_memory_utilization: float
    torch_compile: bool
    cache_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawResult:
    language: str | None  # canonical English name, e.g. "Chinese"
    text: str


class Qwen3Backend:
    """Backend contract: batch transcription of 16 kHz mono float32 arrays."""

    name = ""

    def load(self, p: LoadParams) -> dict:  # returns the kwargs actually passed (for logging)
        raise NotImplementedError

    def transcribe(self, wavs: list[np.ndarray], contexts: list[str], languages: list[str | None]) -> list[RawResult]:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def set_batch_size(self, n: int) -> None:
        pass


# ------------------------------------------------------------------------------------ availability
@dataclass
class BackendStatus:
    available: bool
    reason: str = ""
    install: str = ""
    version: str = ""


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _ver_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split(".")[:3]:
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def backend_status(name: str) -> BackendStatus:
    if name == "qwen_asr":
        v = _version("qwen-asr")
        if not v or importlib.util.find_spec("qwen_asr") is None:
            return BackendStatus(False, "未安装 qwen-asr", "uv sync   （默认依赖组 qwen 即包含 qwen-asr）")
        tv = _version("transformers") or ""
        if tv and _ver_tuple(tv)[:1] >= (5,):
            return BackendStatus(False, f"qwen-asr 需要 transformers 4.57.x，当前为 {tv}",
                                 "uv sync --group qwen --no-group hf", v)
        return BackendStatus(True, version=v)
    if name == "transformers":
        tv = _version("transformers")
        if not tv:
            return BackendStatus(False, "未安装 transformers", "uv sync --no-group qwen --group hf")
        if _ver_tuple(tv)[:2] < MIN_TRANSFORMERS_NATIVE:
            return BackendStatus(
                False,
                f"Transformers 原生 Qwen3-ASR 需要 >= 5.13（当前 {tv}；qwen-asr 固定了 4.57.x，两者不能装在同一环境）",
                "uv sync --no-group qwen --group hf",
                tv,
            )
        return BackendStatus(True, version=tv)
    if name == "vllm":
        if not sys.platform.startswith("linux"):
            return BackendStatus(False, "vLLM 仅支持 Linux（Windows 请使用 WSL2）", "")
        if not _version("vllm"):
            return BackendStatus(False, "未安装 vLLM", "uv sync --extra vllm")
        if backend_status("qwen_asr").available is False:
            return BackendStatus(False, "vLLM 后端通过 qwen-asr 调用，需同时安装 qwen-asr", "uv sync --extra vllm")
        return BackendStatus(True, version=_version("vllm") or "")
    return BackendStatus(False, f"未知后端 {name}")


def aligner_available_for_backend(name: str) -> BackendStatus:
    # Native transformers ships Qwen3ASRForTokenClassification (+ processor.prepare_forced_aligner_inputs)
    # since 5.13, and qwen-asr ships Qwen3ForcedAligner, so every backend supports alignment.
    return backend_status("qwen_asr" if name == "vllm" else name)

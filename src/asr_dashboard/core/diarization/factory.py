"""Backend availability detection and ``auto`` selection (pyannote -> diarizen -> embedding)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from asr_dashboard.config.schema import DiarizationSection
from asr_dashboard.core.diarization import diarizen_backend, pyannote_backend
from asr_dashboard.core.diarization.base import BaseDiarizer


@dataclass
class BackendInfo:
    name: str
    label: str
    available: bool
    reason: str = ""


def _embedding_ok() -> tuple[bool, str]:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return False, "需要 torch 和 transformers"
    return True, ""


def backend_infos(cfg: DiarizationSection) -> list[BackendInfo]:
    p_ok, p_reason = pyannote_backend.availability(cfg.pyannote.model_dir or None)
    d_ok, d_reason = diarizen_backend.availability(cfg.diarizen.env_dir or None, cfg.diarizen.license_accepted)
    e_ok, e_reason = _embedding_ok()
    return [
        BackendInfo("pyannote", "pyannote Community-1（默认，CC-BY-4.0）", p_ok, p_reason),
        BackendInfo("diarizen", "DiariZen（隔离环境，权重仅限非商业用途）", d_ok, d_reason),
        BackendInfo("embedding", "WavLM 声纹 + 聚类（兜底）", e_ok, e_reason),
    ]


def choose_backend(cfg: DiarizationSection) -> BackendInfo:
    infos = {i.name: i for i in backend_infos(cfg)}
    if cfg.backend != "auto":
        return infos[cfg.backend]
    for name in ("pyannote", "diarizen", "embedding"):
        if infos[name].available:
            return infos[name]
    reasons = "；".join(f"{i.name}: {i.reason}" for i in infos.values())
    return BackendInfo("none", "无可用后端", False, reasons)


def resolve_device(cfg: DiarizationSection, qwen_device: str) -> str:
    from asr_dashboard.core.devices import resolve_device as rd

    if cfg.device == "follow":
        return rd(qwen_device)
    return rd(cfg.device)


def create(cfg: DiarizationSection, device: str) -> BaseDiarizer:
    info = choose_backend(cfg)
    if not info.available:
        raise RuntimeError(f"说话人分离后端不可用：{info.reason}")
    if info.name == "pyannote":
        return pyannote_backend.PyannoteDiarizer(cfg.pyannote.model_dir or None, device)
    if info.name == "diarizen":
        return diarizen_backend.DiarizenDiarizer(cfg.diarizen.env_dir or None, cfg.diarizen.model_id, device)
    from asr_dashboard.core.diarization.embedding_backend import EmbeddingDiarizer

    e = cfg.embedding
    return EmbeddingDiarizer(e.model, device, e.clustering, e.similarity_threshold, e.window, e.step)

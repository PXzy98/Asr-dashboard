"""Engine registry: engine id -> factory + settings schema.

Local engine ids: ``qwen3-local`` (variant taken from ``qwen3.variant``). The UI shows the two variants as
``qwen3-local:1.7b`` / ``qwen3-local:0.6b``; selecting one sets ``qwen3.variant``.
API engines use the id of their ``api_models`` entry.
"""

from __future__ import annotations

from dataclasses import dataclass

from asr_dashboard.config.schema import AppConfig
from asr_dashboard.engines.base import BaseASREngine
from asr_dashboard.settings_fields import SettingField

LOCAL_ID = "qwen3-local"


@dataclass
class EngineEntry:
    id: str  # value stored in engine.default / home page combo
    label: str
    group: str  # "local" / "api"
    variant: str | None = None


def list_engines(cfg: AppConfig) -> list[EngineEntry]:
    out = [EngineEntry(f"{LOCAL_ID}:1.7b", "Qwen3-ASR 1.7B", "local", "1.7b"),
           EngineEntry(f"{LOCAL_ID}:0.6b", "Qwen3-ASR 0.6B", "local", "0.6b")]
    for m in cfg.api_models:
        out.append(EngineEntry(m.id, m.name, "api"))
    return out


def split_id(engine_id: str) -> tuple[str, str | None]:
    if engine_id.startswith(LOCAL_ID):
        _, _, var = engine_id.partition(":")
        return LOCAL_ID, var or None
    return engine_id, None


def settings_schema(engine_id: str, cfg: AppConfig) -> list[SettingField]:
    base, _ = split_id(engine_id)
    if base == LOCAL_ID:
        from asr_dashboard.engines.qwen3.engine import Qwen3LocalEngine

        return Qwen3LocalEngine.settings_schema()
    m = cfg.api_model(base)
    if m is None:
        return []
    from asr_dashboard.engines.api import provider_class

    return provider_class(m.type).settings_schema()


def create_engine(engine_id: str, cfg: AppConfig, secrets=None) -> BaseASREngine:
    base, variant = split_id(engine_id)
    if base == LOCAL_ID:
        from asr_dashboard.engines.qwen3.engine import Qwen3LocalEngine

        q = cfg.qwen3.model_copy(deep=True)
        if variant:
            q.variant = variant  # type: ignore[assignment]
        return Qwen3LocalEngine(q, cfg.segmentation.max_sentence_duration)
    m = cfg.api_model(base)
    if m is None:
        raise KeyError(f"找不到引擎 {engine_id}")
    from asr_dashboard.config.secrets import api_key_name
    from asr_dashboard.engines.api import create_api_engine

    key = secrets.get(api_key_name(m.id)) if secrets else ""
    return create_api_engine(m, key, temp_dir=cfg.app.temp_dir or None)

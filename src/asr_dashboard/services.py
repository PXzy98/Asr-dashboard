"""Application services shared by UI pages and workers (engine / diarizer caches, stores)."""

from __future__ import annotations

import logging
import threading

from asr_dashboard.config import secrets as sec
from asr_dashboard.config.manager import ConfigManager
from asr_dashboard.config.schema import AppConfig
from asr_dashboard.core.history import HistoryStore
from asr_dashboard.core.pipeline import Pipeline
from asr_dashboard.engines import registry
from asr_dashboard.engines.base import BaseASREngine

log = logging.getLogger(__name__)


class Services:
    def __init__(self, config: ConfigManager | None = None) -> None:
        self.config = config or ConfigManager()
        self.secrets = self.config.secrets
        self.history = HistoryStore()
        self._engines: dict[str, BaseASREngine] = {}
        self._diarizers: dict[tuple, object] = {}
        self._embedders: dict[tuple, object] = {}
        self._vp_store = None
        self._lock = threading.RLock()

    @property
    def cfg(self) -> AppConfig:
        return self.config.config

    # ---- engines --------------------------------------------------------------------
    def engine(self, engine_id: str, cfg: AppConfig | None = None) -> BaseASREngine:
        """Cached engine instance; local engines keep their loaded model across tasks."""
        cfg = cfg or self.cfg
        base, variant = registry.split_id(engine_id)
        with self._lock:
            if base == registry.LOCAL_ID:
                eng = self._engines.get(base)
                q = cfg.qwen3.model_copy(deep=True)
                if variant:
                    q.variant = variant  # type: ignore[assignment]
                if eng is None:
                    eng = registry.create_engine(engine_id, cfg, self.secrets)
                    self._engines[base] = eng
                else:
                    eng.update_config(q)  # type: ignore[attr-defined]
                    eng.sentence_max_duration = cfg.segmentation.max_sentence_duration  # type: ignore[attr-defined]
                return eng
            # API engines are cheap to build; rebuild to pick up edited settings / keys
            return registry.create_engine(engine_id, cfg, self.secrets)

    def unload_all(self) -> None:
        with self._lock:
            for e in self._engines.values():
                try:
                    e.unload()
                except Exception:  # noqa: BLE001
                    log.exception("unload failed")
            for d in self._diarizers.values():
                try:
                    d.unload()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
            self._diarizers.clear()
            self._embedders.clear()

    # ---- diarization / voiceprints -----------------------------------------------------
    def diarizer(self, cfg: AppConfig):
        from asr_dashboard.core.diarization import factory

        dcfg = cfg.diarization
        device = factory.resolve_device(dcfg, cfg.qwen3.device)
        info = factory.choose_backend(dcfg)
        key = (info.name, device, dcfg.pyannote.model_dir, dcfg.diarizen.env_dir, dcfg.diarizen.model_id,
               dcfg.embedding.model_dump_json())
        with self._lock:
            if key not in self._diarizers:
                self._diarizers[key] = factory.create(dcfg, device)
            return self._diarizers[key]

    def embedder(self, cfg: AppConfig):
        from asr_dashboard.core.diarization import factory
        from asr_dashboard.core.voiceprints.embedder import VoiceprintEmbedder

        device = factory.resolve_device(cfg.diarization, cfg.qwen3.device)
        kind = cfg.voiceprint.embedding_model
        key = (kind, device, cfg.diarization.pyannote.model_dir, cfg.diarization.embedding.model)
        with self._lock:
            if key not in self._embedders:
                shared = None
                if kind == "pyannote":  # reuse the loaded diarization pipeline if any
                    shared = next((d for k, d in self._diarizers.items() if k[0] == "pyannote" and k[1] == device), None)
                self._embedders[key] = VoiceprintEmbedder(kind, device, cfg.diarization.pyannote.model_dir or None,
                                                          cfg.diarization.embedding.model, shared)
            return self._embedders[key]

    @property
    def voiceprints(self):
        if self._vp_store is None:
            from asr_dashboard.core.voiceprints.store import VoiceprintStore

            self._vp_store = VoiceprintStore()
        return self._vp_store

    # ---- pipeline -----------------------------------------------------------------------
    def pipeline(self, engine_id: str, cfg: AppConfig | None = None, keep_loaded: bool | None = None) -> Pipeline:
        cfg = cfg or self.cfg
        eng = self.engine(engine_id, cfg)
        llm_client = None
        if cfg.llm_active:
            from asr_dashboard.text.llm_client import LLMClient

            llm_client = LLMClient(cfg.llm.base_url, self.secrets.get(sec.LLM_KEY), cfg.llm.model,
                                   timeout=cfg.llm.timeout, json_mode=cfg.llm.json_mode)
        return Pipeline(cfg, eng, secrets=self.secrets, diarizer_provider=self.diarizer,
                        voiceprint_store=self.voiceprints, embedder_provider=self.embedder, llm_client=llm_client,
                        keep_engine_loaded=keep_loaded)

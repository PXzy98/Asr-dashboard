"""Read / write ``config.yaml`` atomically and give access to the secret store."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

import yaml

from asr_dashboard import paths
from asr_dashboard.config import secrets as sec
from asr_dashboard.config.schema import AppConfig, load_config

log = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, path: Path | None = None, secret_store: sec.SecretStore | None = None) -> None:
        self.path = Path(path) if path else paths.default_config_path()
        self.secrets = secret_store or sec.SecretStore()
        self._lock = threading.RLock()
        self._listeners: list[Callable[[AppConfig], None]] = []
        self.problems: list[str] = []
        self.config = self.load()

    def load(self) -> AppConfig:
        data = None
        if self.path.exists():
            try:
                data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                self.problems = [f"无法解析 {self.path}：{e}，已使用默认配置"]
                log.warning(self.problems[0])
                return AppConfig()
        else:
            log.info("配置文件 %s 不存在，使用默认配置", self.path)
        cfg, self.problems = load_config(data)
        names = [sec.LLM_KEY, sec.REALTIME_KEY, sec.HF_TOKEN] + [sec.api_key_name(m.id) for m in cfg.api_models]
        try:
            self.secrets.preload(names)
        except Exception as e:  # noqa: BLE001
            log.warning("预加载密钥失败：%s", e)
        return cfg

    def save(self, cfg: AppConfig | None = None) -> None:
        with self._lock:
            if cfg is not None:
                self.config = cfg
            data = self.config.model_dump(mode="json")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            os.replace(tmp, self.path)
        log.debug("配置已保存到 %s", self.path)
        for cb in list(self._listeners):
            try:
                cb(self.config)
            except Exception:  # noqa: BLE001
                log.exception("config listener failed")

    def update(self, mutator: Callable[[AppConfig], None]) -> AppConfig:
        """Apply ``mutator`` on a validated copy; persists only if the result is valid."""
        with self._lock:
            new = self.config.model_copy(deep=True)
            mutator(new)
            new = AppConfig.model_validate(new.model_dump())
            self.save(new)
            return new

    def reset_defaults(self, keep_api_models: bool = True) -> AppConfig:
        new = AppConfig()
        if keep_api_models:
            new.api_models = self.config.api_models
            new.glossaries = self.config.glossaries
        self.save(new)
        return new

    def subscribe(self, cb: Callable[[AppConfig], None]) -> None:
        self._listeners.append(cb)

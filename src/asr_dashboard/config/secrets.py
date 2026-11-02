"""Secret storage: system keyring first, plaintext ``~/.asr-dashboard/secrets.yaml`` as fallback."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import yaml

from asr_dashboard import paths
from asr_dashboard.redact import register_secret

log = logging.getLogger(__name__)
SERVICE = "asr-dashboard"


# well-known secret names
def api_key_name(model_id: str) -> str:
    return f"api_model:{model_id}"


LLM_KEY = "llm"
REALTIME_KEY = "realtime"
HF_TOKEN = "hf_token"


class SecretStore:
    def __init__(self, file_path: Path | None = None, use_keyring: bool | None = None) -> None:
        self.file_path = file_path or paths.secrets_file()
        self._lock = threading.Lock()
        if use_keyring is None:
            use_keyring = os.environ.get("ASR_DASHBOARD_NO_KEYRING") != "1"
        self._keyring = self._probe_keyring() if use_keyring else None

    @staticmethod
    def _probe_keyring():
        try:
            import keyring
            from keyring.backends import fail

            kr = keyring.get_keyring()
            if isinstance(kr, fail.Keyring) or "fail" in type(kr).__module__ or "null" in type(kr).__module__:
                return None
            # some backends only fail on first use
            keyring.get_password(SERVICE, "__probe__")
            return keyring
        except Exception as e:  # noqa: BLE001
            log.info("系统钥匙串不可用，改用明文文件存储密钥：%s", e)
            return None

    @property
    def is_plaintext(self) -> bool:
        return self._keyring is None

    @property
    def backend_name(self) -> str:
        if self._keyring is None:
            return f"明文文件 {self.file_path}"
        return type(self._keyring.get_keyring()).__name__

    # ---- file backend -------------------------------------------------
    def _read_file(self) -> dict:
        if not self.file_path.exists():
            return {}
        try:
            return yaml.safe_load(self.file_path.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001
            log.warning("读取 secrets.yaml 失败：%s", e)
            return {}

    def _write_file(self, data: dict) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file_path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        os.replace(tmp, self.file_path)
        try:
            os.chmod(self.file_path, 0o600)
        except OSError:
            pass

    # ---- public API ---------------------------------------------------
    def get(self, name: str) -> str:
        with self._lock:
            value = ""
            if self._keyring is not None:
                try:
                    value = self._keyring.get_password(SERVICE, name) or ""
                except Exception as e:  # noqa: BLE001
                    log.warning("钥匙串读取失败：%s", e)
            if not value:
                value = str(self._read_file().get(name) or "")
        register_secret(value)
        return value

    def set(self, name: str, value: str) -> None:
        register_secret(value)
        with self._lock:
            if not value:
                self._delete_locked(name)
                return
            if self._keyring is not None:
                try:
                    self._keyring.set_password(SERVICE, name, value)
                    data = self._read_file()
                    if name in data:  # don't leave a stale plaintext copy
                        data.pop(name)
                        self._write_file(data)
                    return
                except Exception as e:  # noqa: BLE001
                    log.warning("钥匙串写入失败，改用明文文件：%s", e)
            data = self._read_file()
            data[name] = value
            self._write_file(data)

    def delete(self, name: str) -> None:
        with self._lock:
            self._delete_locked(name)

    def _delete_locked(self, name: str) -> None:
        if self._keyring is not None:
            try:
                self._keyring.delete_password(SERVICE, name)
            except Exception:  # noqa: BLE001
                pass
        data = self._read_file()
        if name in data:
            data.pop(name)
            self._write_file(data)

    def preload(self, names: list[str]) -> None:
        """Register all known secrets with the redactor so they never leak into logs."""
        for n in names:
            self.get(n)

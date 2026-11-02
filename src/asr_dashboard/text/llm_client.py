"""OpenAI-compatible ``/chat/completions`` client and the save-time validation of LLM settings."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from asr_dashboard.models import CancelledError, CancelToken
from asr_dashboard.redact import redact, register_secret


class LLMError(RuntimeError):
    def __init__(self, message: str, kind: str = "error") -> None:
        super().__init__(redact(message))
        self.kind = kind


def is_local(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".local")


def strip_fences(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    # tolerate <think>...</think> blocks of reasoning models
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    return t


def parse_json(text: str):
    t = strip_fences(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # last resort: first {...} or [...] block
        m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
        if m:
            return json.loads(m.group(1))
        raise


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120, json_mode: bool = False,
                 temperature: float = 0.2, transport: httpx.BaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        register_secret(self.api_key)
        self.model = model
        self.timeout = timeout
        self.json_mode = json_mode
        self.temperature = temperature
        self.transport = transport

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(self, messages: list[dict], want_json: bool = True, temperature: float | None = None,
             cancel: CancelToken | None = None, timeout: float | None = None, json_mode: bool | None = None) -> str:
        body: dict = {"model": self.model, "messages": messages,
                      "temperature": self.temperature if temperature is None else temperature}
        if want_json and (self.json_mode if json_mode is None else json_mode):
            body["response_format"] = {"type": "json_object"}
        client = httpx.Client(timeout=timeout or self.timeout, transport=self.transport)
        if cancel:
            cancel.on_cancel(client.close)
        try:
            resp = client.post(f"{self.base_url}/chat/completions", headers=self._headers(), content=json.dumps(body))
        except (httpx.TimeoutException, httpx.TransportError) as e:
            if cancel and cancel.cancelled:
                raise CancelledError("任务已取消") from e
            raise LLMError(f"无法连接到 {self.base_url}：{e}", "network") from e
        finally:
            if cancel:
                cancel.remove_callback(client.close)
            client.close()
        if resp.status_code in (401, 403):
            raise LLMError(f"认证失败（HTTP {resp.status_code}）：{resp.text[:300]}", "auth")
        if resp.status_code == 404 or _model_missing(resp.text):
            raise LLMError(f"模型不存在或地址错误（HTTP {resp.status_code}）：{resp.text[:300]}", "model")
        if resp.status_code >= 400:
            kind = "json_mode" if "response_format" in resp.text or "json_object" in resp.text else "http"
            raise LLMError(f"HTTP {resp.status_code}：{resp.text[:300]}", kind)
        try:
            return resp.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"返回内容无法解析：{resp.text[:300]}", "parse") from e

    def chat_json(self, messages: list[dict], cancel: CancelToken | None = None, temperature: float | None = None):
        content = self.chat(messages, True, temperature, cancel)
        try:
            return parse_json(content)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"模型返回的不是合法 JSON：{content[:200]}", "parse") from e


def _model_missing(text: str) -> bool:
    t = text.lower()
    return "model" in t and any(k in t for k in ("not found", "does not exist", "not exist", "no such model",
                                                  "model_not_found", "unknown model", "invalid model"))


@dataclass
class ValidationResult:
    ok: bool
    kind: str = ""  # network / auth / model / parse / http
    message: str = ""
    latency: float = 0.0
    json_mode: bool = False

    @property
    def hint(self) -> str:
        return {
            "network": "请检查 API 地址是否正确（通常以 /v1 结尾）、服务是否已启动、网络或代理是否可用。",
            "auth": "API Key 无效或没有权限，请检查 Key 是否填写正确、是否过期。",
            "model": "模型名不存在，请检查拼写，或在服务商控制台确认可用的模型名。",
            "parse": "服务返回的内容无法解析，确认这是 OpenAI 兼容的 /chat/completions 接口。",
            "http": "服务返回错误，请查看错误详情。",
        }.get(self.kind, "")


PROBE = [{"role": "system", "content": "You are a health check. Reply with JSON only."},
         {"role": "user", "content": 'Return exactly this JSON and nothing else: {"ok": true}'}]


def validate(base_url: str, api_key: str, model: str, transport: httpx.BaseTransport | None = None,
             timeout: float = 20.0) -> ValidationResult:
    if not base_url.strip() or not model.strip():
        return ValidationResult(False, "http", "请填写 API 地址和模型名")
    c = LLMClient(base_url, api_key, model, timeout=timeout, json_mode=True, temperature=0.0, transport=transport)
    t0 = time.perf_counter()
    json_mode = True
    try:
        try:
            content = c.chat(PROBE, want_json=True)
        except LLMError as e:
            if e.kind not in ("json_mode", "http"):
                raise
            json_mode = False  # server rejects response_format -> prompt-only JSON
            content = c.chat(PROBE, want_json=True, json_mode=False)
        latency = time.perf_counter() - t0
        try:
            data = parse_json(content)
        except Exception:  # noqa: BLE001
            return ValidationResult(False, "parse", f"返回内容无法解析为 JSON：{content[:200]}", latency, json_mode)
        if not (isinstance(data, dict) and data.get("ok") is True):
            return ValidationResult(False, "parse", f"返回内容不符合预期：{content[:200]}", latency, json_mode)
        return ValidationResult(True, "", "可用", latency, json_mode)
    except LLMError as e:
        return ValidationResult(False, e.kind if e.kind != "json_mode" else "http", str(e), time.perf_counter() - t0)

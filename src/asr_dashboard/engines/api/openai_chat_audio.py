"""``POST {base_url}/chat/completions`` with an ``audio_url`` data-URL (remote Qwen3-ASR on vLLM / qwen-asr-serve).

Request shape verified against the Qwen3-ASR README (``{"type": "audio_url", "audio_url": {"url": ...}}``).
The reply is ``language Chinese<asr_text>...``; the context goes into the system message, exactly as the
qwen-asr package builds its prompt. No timestamps -> the pipeline chunks audio before sending.
"""

from __future__ import annotations

import base64
import json

from asr_dashboard.engines.api.base import APIError, BaseAPIEngine, CallResult, Upload
from asr_dashboard.engines.qwen3.common import lang_code
from asr_dashboard.models import CancelToken, Segment
from asr_dashboard.settings_fields import SettingField

ASR_TAG = "<asr_text>"


def parse_asr_output(raw: str) -> tuple[str | None, str]:
    """Mirror of qwen_asr.parse_asr_output (language name, text)."""
    s = (raw or "").strip()
    if not s:
        return None, ""
    if ASR_TAG not in s:
        return None, s
    meta, text = s.split(ASR_TAG, 1)
    if "language none" in meta.lower():
        return None, text.strip()
    lang = None
    for line in meta.splitlines():
        line = line.strip()
        if line.lower().startswith("language "):
            val = line[len("language "):].strip()
            lang = val[:1].upper() + val[1:].lower() if val else None
            break
    return lang, text.strip()


class OpenAIChatAudioEngine(BaseAPIEngine):
    provider_type = "openai_chat_audio"
    provider_label = "OpenAI 兼容 Chat 音频输入（远程 Qwen3-ASR）"

    def provider_timestamps(self) -> bool:
        return False

    @classmethod
    def provider_fields(cls) -> list[SettingField]:
        return [SettingField("options.max_tokens", "max_tokens", "int", 4096, minimum=64, maximum=32768)]

    def call(self, up: Upload, language: str | None, context: str | None, cancel: CancelToken) -> CallResult:
        b64 = base64.b64encode(up.data).decode()
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": f"data:{up.mime};base64,{b64}"}}]})
        body = {"messages": messages, "temperature": 0.0, "max_tokens": int(self.cfg.options.get("max_tokens", 4096))}
        if self.cfg.model:
            body["model"] = self.cfg.model
        body.update(self.cfg.extra or {})
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        resp = self.request("POST", url, cancel, headers={**self.auth_headers(), "Content-Type": "application/json"},
                            content=json.dumps(body))
        try:
            d = resp.json()
            content = d["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise APIError(f"无法解析返回内容：{resp.text[:300]}", kind="parse") from e
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        lang, text = parse_asr_output(content)
        code = lang_code(lang)
        return CallResult(d, [Segment(text, 0.0, up.duration, language=code)] if text else [], False, code)

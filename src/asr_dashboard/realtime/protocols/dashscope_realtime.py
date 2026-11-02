"""Alibaba Cloud Model Studio realtime ASR (``qwen3-asr-flash-realtime``) over the OpenAI-realtime-style API.

From the dashscope 2.0.3 SDK source: endpoint ``wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=...``
(``dashscope-intl`` for the international site), ``Authorization: Bearer <key>``, ``session.update`` with
``input_audio_format``/``sample_rate``/``input_audio_transcription`` (``language``, ``corpus.text``) and
``turn_detection`` (server_vad), ``input_audio_buffer.append``, ``session.finish`` -> ``session.finished``;
final text arrives as ``conversation.item.input_audio_transcription.completed`` (``transcript``).
The partial-result event name (``...transcription.text`` with ``text``+``stash``) is not present in the SDK
source and could not be checked online; it is handled defensively together with ``*.delta``.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlencode, urlsplit

from asr_dashboard.realtime.protocols.base import BaseRealtimeClient, RealtimeError, b64_pcm

DEFAULT_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_MODEL = "qwen3-asr-flash-realtime"


class DashScopeRealtimeClient(BaseRealtimeClient):
    protocol = "dashscope_realtime"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.model = self.model or DEFAULT_MODEL
        self._error: RealtimeError | None = None

    def ws_url(self) -> str:
        url = self.url or DEFAULT_URL
        if "model=" in (urlsplit(url).query or ""):
            return url
        return url + ("&" if "?" in url else "?") + urlencode({"model": self.model})

    async def handshake(self) -> None:
        msg = await self._recv(10.0)
        if msg.get("type") == "error":
            raise self._to_error(msg)
        if msg.get("type") != "session.created":
            raise RealtimeError(f"未收到 session.created（收到 {msg.get('type')}）", "protocol")
        tr: dict = {}
        if self.language:
            tr["language"] = self.language
        if self.context:
            tr["corpus"] = {"text": self.context}
        await self._send({"type": "session.update", "session": {
            "modalities": ["text"], "input_audio_format": "pcm", "sample_rate": 16000,
            "input_audio_transcription": tr,
            "turn_detection": {"type": "server_vad", "threshold": 0.2, "silence_duration_ms": 800},
        }})
        msg = await self._recv(10.0)
        if msg.get("type") == "error":
            raise self._to_error(msg)

    def _to_error(self, msg: dict) -> RealtimeError:
        err = msg.get("error") or {}
        text = err.get("message") if isinstance(err, dict) else str(err)
        code = (err.get("code") if isinstance(err, dict) else "") or ""
        low = f"{code} {text}".lower()
        kind = "auth" if ("auth" in low or "apikey" in low or "api key" in low) else \
            "model" if "model" in low else "server"
        return RealtimeError(f"{code} {text}".strip(), kind)

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": b64_pcm(pcm16)})

    def handle(self, msg: dict) -> None:
        t = msg.get("type") or ""
        if t.endswith("input_audio_transcription.completed"):
            text = (msg.get("transcript") or msg.get("text") or "").strip()
            if text:
                self._text_seen()
                self.events.on_final(text)
            self.events.on_partial("")
        elif t.endswith("input_audio_transcription.text") or t.endswith("input_audio_transcription.delta"):
            partial = (msg.get("text") or "") + (msg.get("stash") or "") or msg.get("delta") or ""
            if partial:
                self._text_seen()
                self.events.on_partial(partial)
        elif t == "session.finished":
            self._done.set()
        elif t == "error":
            self._error = self._to_error(msg)
            self.events.on_error(self._error)

    async def finish(self, timeout: float = 10.0) -> None:
        await self._send({"type": "session.finish"})
        try:
            await asyncio.wait_for(self._done.wait(), timeout)
        except TimeoutError as e:
            raise RealtimeError("等待 session.finished 超时", "timeout") from e
        if self._error:
            raise self._error


def make_client(protocol: str, url: str, api_key: str, model: str, language: str | None, context: str = "",
                connect=None) -> BaseRealtimeClient:
    if protocol == "dashscope_realtime":
        return DashScopeRealtimeClient(url, api_key, model, language, context, connect)
    from asr_dashboard.realtime.protocols.vllm_realtime import VllmRealtimeClient

    return VllmRealtimeClient(url, api_key, model, language, context, connect)

"""vLLM ``/v1/realtime`` protocol (verified against vLLM 0.32 ``entrypoints/speech_to_text/realtime``).

client -> server: ``session.update {model}``, ``input_audio_buffer.commit`` (starts generation),
``input_audio_buffer.append {audio: base64 PCM16 16 kHz mono}``, ``input_audio_buffer.commit {final: true}``.
server -> client: ``session.created``, ``transcription.delta {delta}``, ``transcription.done {text}``,
``error {error, code}``.
``session.update`` only carries ``model`` – vLLM ignores prompt/language here, so context cannot be used.
The server only exposes the endpoint when the model runs with a realtime architecture
(``Qwen3ASRRealtimeGeneration``, see server/start_realtime.sh).
"""

from __future__ import annotations

import asyncio

from asr_dashboard.realtime.protocols.base import BaseRealtimeClient, RealtimeError, b64_pcm

SENT_END = "。！？!?.…；;"


class SentenceSplitter:
    """Turns a growing transcript into confirmed sentences + an unconfirmed tail."""

    def __init__(self) -> None:
        self.confirmed = 0  # chars already emitted as final

    def feed(self, full: str) -> tuple[list[str], str]:
        tail = full[self.confirmed:]
        finals = []
        start = 0
        for i, ch in enumerate(tail):
            if ch in SENT_END and not (ch == "." and i + 1 < len(tail) and tail[i + 1].isdigit()):
                piece = tail[start:i + 1].strip()
                if piece:
                    finals.append(piece)
                start = i + 1
        self.confirmed += start
        return finals, tail[start:].strip()

    def flush(self, full: str) -> str:
        rest = full[self.confirmed:].strip()
        self.confirmed = len(full)
        return rest


class VllmRealtimeClient(BaseRealtimeClient):
    protocol = "vllm_realtime"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.context_supported = False
        self.text = ""
        self.splitter = SentenceSplitter()
        self._error: RealtimeError | None = None

    async def handshake(self) -> None:
        msg = await self._recv(10.0)
        if msg.get("type") != "session.created":
            raise RealtimeError(f"未收到 session.created（收到 {msg.get('type')}），可能不是 vLLM realtime 接口",
                                "protocol")
        await self._send({"type": "session.update", "model": self.model})
        # vLLM answers session.update only on error; give it a moment to complain about the model
        try:
            msg = await self._recv(0.8)
            if msg.get("type") == "error":
                code = msg.get("code") or ""
                kind = "model" if "model" in code else "server"
                raise RealtimeError(f"{msg.get('error')}", kind)
        except RealtimeError as e:
            if e.kind != "timeout":
                raise
        await self._send({"type": "input_audio_buffer.commit"})  # start generation

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": b64_pcm(pcm16)})

    def handle(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "transcription.delta":
            delta = msg.get("delta") or ""
            if delta:
                self._text_seen()
                self.text += delta
                finals, pending = self.splitter.feed(self.text)
                for f in finals:
                    self.events.on_final(f)
                self.events.on_partial(pending)
        elif t == "transcription.done":
            self.text = msg.get("text") or self.text
            rest = self.splitter.flush(self.text)
            if rest:
                self._text_seen()
                self.events.on_final(rest)
            self.events.on_partial("")
            self._done.set()
        elif t == "error":
            code = msg.get("code") or ""
            kind = "model" if "model" in code else "server"
            self._error = RealtimeError(str(msg.get("error")), kind)
            self.events.on_error(self._error)

    async def finish(self, timeout: float = 10.0) -> None:
        await self._send({"type": "input_audio_buffer.commit", "final": True})
        try:
            await asyncio.wait_for(self._done.wait(), timeout)
        except TimeoutError as e:
            raise RealtimeError("等待最终识别结果超时", "timeout") from e
        if self._error:
            raise self._error

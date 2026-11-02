"""Realtime (streaming) ASR client interface over WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from asr_dashboard.redact import redact, register_secret

log = logging.getLogger(__name__)


class RealtimeError(RuntimeError):
    """kind: connect / auth / model / protocol / timeout / server"""

    def __init__(self, message: str, kind: str = "server") -> None:
        super().__init__(redact(message))
        self.kind = kind


KIND_HINT = {
    "connect": "无法连接：检查服务地址（ws:// 或 wss://）、端口、防火墙，以及服务是否已启动",
    "auth": "认证失败：检查 API Key",
    "model": "模型名不匹配：模型名需与服务端加载的模型一致",
    "protocol": "协议不匹配：服务端不是所选协议的实时接口（vLLM 需要以 realtime 架构启动，见 README）",
    "timeout": "超时：服务端在规定时间内没有返回识别结果",
    "server": "服务端返回错误",
}


@dataclass
class RealtimeEvents:
    on_partial: Callable[[str], None] = lambda t: None  # full current unconfirmed text
    on_final: Callable[[str], None] = lambda t: None  # one confirmed piece of text
    on_error: Callable[[RealtimeError], None] = lambda e: None
    on_first_text: Callable[[], None] = lambda: None


def b64_pcm(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode()


class BaseRealtimeClient:
    protocol = ""

    def __init__(self, url: str, api_key: str = "", model: str = "", language: str | None = None,
                 context: str = "", connect=None) -> None:
        self.url = url
        self.api_key = api_key or ""
        register_secret(self.api_key)
        self.model = model
        self.language = None if language in (None, "", "auto") else language
        self.context = context
        self._connect = connect
        self.ws = None
        self.events = RealtimeEvents()
        self._recv_task: asyncio.Task | None = None
        self._done = asyncio.Event()
        self._got_text = False
        self.context_supported = True

    # ---- connection --------------------------------------------------------------
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def ws_url(self) -> str:
        return self.url

    async def _open(self, timeout: float = 10.0):
        import websockets

        connect = self._connect or websockets.connect
        try:
            return await asyncio.wait_for(
                connect(self.ws_url(), additional_headers=self.headers(), max_size=None, open_timeout=timeout),
                timeout + 1)
        except TimeoutError as e:
            raise RealtimeError(f"连接 {self.url} 超时", "connect") from e
        except Exception as e:  # noqa: BLE001
            status = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
            if status in (401, 403):
                raise RealtimeError(f"认证失败（HTTP {status}）", "auth") from e
            if status in (404, 400, 405, 426):
                raise RealtimeError(f"服务端拒绝 WebSocket 握手（HTTP {status}），可能不是实时接口", "protocol") from e
            raise RealtimeError(f"无法连接 {self.url}：{e}", "connect") from e

    async def connect(self, events: RealtimeEvents) -> None:
        self.events = events
        self._done = asyncio.Event()
        self.ws = await self._open()
        try:
            await self.handshake()
        except Exception:
            await self.close()
            raise
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv(self, timeout: float) -> dict:
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout)
        except TimeoutError as e:
            raise RealtimeError("等待服务端事件超时", "timeout") from e
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError) as e:
            raise RealtimeError(f"服务端返回非 JSON 数据：{str(raw)[:100]}", "protocol") from e

    async def _send(self, obj: dict) -> None:
        await self.ws.send(json.dumps(obj))

    async def _recv_loop(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                self.handle(msg)
        except Exception as e:  # noqa: BLE001 - connection drop
            if not self._done.is_set():
                self.events.on_error(RealtimeError(f"连接中断：{e}", "connect"))
        finally:
            self._done.set()

    def _text_seen(self) -> None:
        if not self._got_text:
            self._got_text = True
            self.events.on_first_text()

    async def close(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except BaseException:  # noqa: BLE001
                pass
            self._recv_task = None

    @property
    def closed(self) -> bool:
        return self.ws is None or self._done.is_set()

    # ---- protocol-specific -------------------------------------------------------
    async def handshake(self) -> None:
        raise NotImplementedError

    async def send_audio(self, pcm16: bytes) -> None:
        raise NotImplementedError

    async def finish(self, timeout: float = 10.0) -> None:
        """Signal end of audio and wait for the final result."""
        raise NotImplementedError

    def handle(self, msg: dict) -> None:
        raise NotImplementedError

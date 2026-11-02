"""One realtime session: capture -> send -> assemble text -> record wav + txt/srt -> (refine afterwards).

Runs an asyncio loop in its own thread; the UI feeds 16 kHz float32 frames with ``feed()``.
Reconnects automatically; while disconnected audio is buffered (max 60 s) and replayed after reconnecting.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asr_dashboard.core.audio import SAMPLE_RATE, float_to_pcm16
from asr_dashboard.core.segmentation import strip_leading_tokens, text_overlap_len
from asr_dashboard.export.streaming import StreamingWriter
from asr_dashboard.export.writers import ExportOptions
from asr_dashboard.models import Segment
from asr_dashboard.realtime.protocols import make_client
from asr_dashboard.realtime.protocols.base import RealtimeError, RealtimeEvents

log = logging.getLogger(__name__)

MAX_BUFFER_SECONDS = 60.0
REPLAY_SECONDS = 0.5  # audio re-sent after a drop (may have been lost in flight)


@dataclass
class SessionCallbacks:
    on_partial: Callable[[str], None] = lambda t: None
    on_final: Callable[[dict], None] = lambda s: None  # {start, end, text, speaker}
    on_status: Callable[[str], None] = lambda s: None
    on_error: Callable[[str], None] = lambda e: None
    on_level: Callable[[float], None] = lambda v: None
    on_speaker: Callable[[int, str], None] = lambda idx, name: None  # late speaker label for sentence idx
    on_stopped: Callable[[dict], None] = lambda info: None


@dataclass
class SessionSettings:
    protocol: str
    url: str
    api_key: str
    model: str
    language: str | None
    chunk_ms: int = 100
    auto_reconnect: bool = True
    max_reconnects: int = 5
    context: str = ""
    connect: object = None  # injectable websocket connect (tests)
    reconnect_delay: float = 1.0


@dataclass
class SessionState:
    sentences: list[dict] = field(default_factory=list)
    partial: str = ""
    captured_samples: int = 0
    sent_samples: int = 0
    reconnects: int = 0
    error: str | None = None


class RealtimeSession:
    def __init__(self, settings: SessionSettings, out_dir: str | Path, stem: str, callbacks: SessionCallbacks | None = None,
                 live_speaker: Callable[[np.ndarray], str | None] | None = None) -> None:
        self.s = settings
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stem = stem
        self.cb = callbacks or SessionCallbacks()
        self.live_speaker = live_speaker
        self.state = SessionState()
        self.wav_path = self.out_dir / f"{stem}.wav"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = False
        self._ready = threading.Event()
        self._audio: list[np.ndarray] = []  # int16 copy for live speaker id
        self._client = None
        self._connected = False
        self._backlog: collections.deque[bytes] = collections.deque()
        self._backlog_samples = 0
        self._recent: collections.deque[bytes] = collections.deque()
        self._dropped_samples = 0
        self._last_final_end = 0.0
        self._dedupe_next = False  # first sentence after a reconnect may repeat replayed audio
        self._writer = StreamingWriter(self.out_dir, stem, ["txt", "srt"], ExportOptions(txt_timestamps=True),
                                       overwrite=True)
        self._wav = None
        self._fatal: RealtimeError | None = None

    # ------------------------------------------------------------------ public (any thread)
    def start(self) -> None:
        import soundfile as sf

        self._wav = sf.SoundFile(str(self.wav_path), "w", SAMPLE_RATE, 1, "PCM_16")
        self._thread = threading.Thread(target=self._thread_main, name="realtime-session", daemon=True)
        self._thread.start()
        self._ready.wait(5)

    def feed(self, frames: np.ndarray) -> None:
        if self._paused or self._stop.is_set() or self._loop is None:
            return
        frames = np.asarray(frames, dtype=np.float32)
        level = float(np.sqrt(np.mean(frames**2))) if len(frames) else 0.0
        self.cb.on_level(level)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, frames)

    def pause(self) -> None:
        self._paused = True
        self.cb.on_status("已暂停")

    def resume(self) -> None:
        self._paused = False
        self.cb.on_status("录音中")

    def stop(self, wait: bool = True, timeout: float = 20.0) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        if wait and self._thread:
            self._thread.join(timeout)

    @property
    def draft_text(self) -> str:
        return "\n".join(s["text"] for s in self.state.sentences)

    # ------------------------------------------------------------------ loop
    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._ready.set()
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:  # noqa: BLE001
            log.exception("实时会话异常")
            self.state.error = str(e)
            self.cb.on_error(str(e))
        finally:
            self._finish_files()
            self._loop.close()
            self.cb.on_stopped({"wav": str(self.wav_path), "txt": str(self._writer.paths.get("txt")),
                                "srt": str(self._writer.paths.get("srt")), "sentences": list(self.state.sentences),
                                "error": self.state.error, "duration": self.state.captured_samples / SAMPLE_RATE})

    def _events(self) -> RealtimeEvents:
        return RealtimeEvents(on_partial=self._on_partial, on_final=self._on_final, on_error=self._on_client_error)

    async def _open_client(self, attempts: int):
        """Try to connect up to ``attempts`` times; returns a client or None (sets _fatal on fatal errors)."""
        for i in range(attempts):
            if self._stop.is_set() and i > 0:
                return None
            client = make_client(self.s.protocol, self.s.url, self.s.api_key, self.s.model, self.s.language,
                                 self.s.context, self.s.connect)
            try:
                await client.connect(self._events())
                return client
            except RealtimeError as e:
                if e.kind in ("auth", "model", "protocol"):
                    self._fatal = e
                    self.cb.on_error(f"{e}")
                    return None
                if i + 1 < attempts:
                    delay = min(30.0, self.s.reconnect_delay * 2 ** i)
                    self.cb.on_status(f"连接失败，{delay:.0f} 秒后重试（{i + 1}/{attempts}）")
                    await asyncio.sleep(delay)
        return None

    async def _main(self) -> None:
        self.cb.on_status("连接中…")
        self._client = await self._open_client(1)
        if self._client is not None:
            self._connected = True
            self.cb.on_status("已连接，录音中")
        step = int(SAMPLE_RATE * self.s.chunk_ms / 1000)
        pending = np.zeros(0, dtype=np.float32)
        reconnect_task: asyncio.Task | None = None
        while True:
            item = await self._queue.get()
            if item is None:
                break
            self._record(item)
            pending = np.concatenate([pending, item])
            while len(pending) >= step:
                await self._send(float_to_pcm16(pending[:step]), step)
                pending = pending[step:]
            if self._fatal is not None:
                break
            lost = not self._connected or (self._client is not None and self._client.closed)
            if lost and self.s.auto_reconnect and (reconnect_task is None or reconnect_task.done()) \
                    and self.state.reconnects < self.s.max_reconnects:
                self._connected = False
                reconnect_task = asyncio.create_task(self._reconnect())
        if reconnect_task is not None and not reconnect_task.done():
            try:
                await asyncio.wait_for(reconnect_task, 15)
            except Exception:  # noqa: BLE001
                reconnect_task.cancel()
        while not self._queue.empty():  # drain leftovers into the recording
            item = self._queue.get_nowait()
            if item is not None:
                self._record(item)
                pending = np.concatenate([pending, item])
        if len(pending) and self._fatal is None:
            await self._send(float_to_pcm16(pending), len(pending))
        lost = not self._connected or (self._client is not None and self._client.closed)
        if lost and self._fatal is None and self.s.auto_reconnect and self.state.reconnects < self.s.max_reconnects:
            await self._reconnect()  # flush the backlog before finishing so no audio is lost
        if self._client is not None and self._connected and self._fatal is None:
            try:
                await self._client.finish(10.0)
            except RealtimeError as e:
                self.cb.on_error(f"结束时出错：{e}")
        if self._client is not None:
            await self._client.close()
        if self._fatal is not None:
            self.state.error = str(self._fatal)
        elif not self._connected and self.state.reconnects:
            self.state.error = self.state.error or "连接中断，录音已在本地保存"

    async def _reconnect(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        self.state.reconnects += 1
        self.cb.on_status(f"连接中断，正在重连（第 {self.state.reconnects} 次）…")
        await asyncio.sleep(self.s.reconnect_delay)
        client = await self._open_client(max(1, self.s.max_reconnects - self.state.reconnects + 1))
        if client is None:
            self.state.error = "重连失败，录音仍在本地保存"
            self.cb.on_error(self.state.error)
            return
        self._client = client
        self._dedupe_next = bool(self.state.sentences)
        if self._dropped_samples:
            self._emit_final(f"[连接中断 {self._dropped_samples / SAMPLE_RATE:.0f} 秒]", marker=True)
            self._dropped_samples = 0
        # replay buffered audio in order; new audio keeps going to the backlog until it is empty
        while self._backlog:
            chunk = self._backlog.popleft()
            n = len(chunk) // 2
            self._backlog_samples -= n
            await client.send_audio(chunk)
            self.state.sent_samples += n
        self._connected = True
        self.cb.on_status("已重新连接，录音中")

    async def _send(self, pcm: bytes, n_samples: int) -> None:
        if self._connected and not self._backlog and self._client is not None and not self._client.closed:
            try:
                await self._client.send_audio(pcm)
                await asyncio.sleep(0)  # let the receive task notice a closed connection promptly
                if not self._client.closed:
                    self.state.sent_samples += n_samples
                    self._recent.append(pcm)
                    while sum(len(x) for x in self._recent) > REPLAY_SECONDS * SAMPLE_RATE * 2:
                        self._recent.popleft()
                    return
            except Exception as e:  # noqa: BLE001
                log.warning("发送音频失败：%s", e)
            self._connected = False
            # audio written just before the drop may never have reached the server: resend it
            self._backlog.extendleft(reversed(self._recent))
            self._backlog_samples += sum(len(x) // 2 for x in self._recent)
            self._recent.clear()
        # disconnected (or replay still running): buffer, bounded to 60 s, dropping the oldest audio
        self._backlog.append(pcm)
        self._backlog_samples += n_samples
        while self._backlog_samples > MAX_BUFFER_SECONDS * SAMPLE_RATE and self._backlog:
            old = self._backlog.popleft()
            self._backlog_samples -= len(old) // 2
            self._dropped_samples += len(old) // 2

    def _record(self, frames: np.ndarray) -> None:
        if self._wav is not None:
            self._wav.write(frames)
            self.state.captured_samples += len(frames)
            if self.state.captured_samples % (SAMPLE_RATE * 2) < len(frames):
                self._wav.flush()  # keep the header valid so a crash leaves a playable file
        if self.live_speaker is not None:
            self._audio.append(float_to_pcm16(frames))

    # ------------------------------------------------------------------ client events (loop thread)
    def _on_partial(self, text: str) -> None:
        self.state.partial = text
        self.cb.on_partial(text)

    def _on_final(self, text: str) -> None:
        if self._dedupe_next:
            self._dedupe_next = False
            prev = next((x["text"] for x in reversed(self.state.sentences) if not x.get("marker")), "")
            n = text_overlap_len(prev, text)  # >= 2 tokens, avoids eating a genuinely repeated word
            if n:
                text = strip_leading_tokens(text, n)
                if not text.strip():
                    return
        self._emit_final(text)

    def _emit_final(self, text: str, marker: bool = False) -> None:
        end = max(self._last_final_end + 0.01, self.state.sent_samples / SAMPLE_RATE)
        sent = {"start": self._last_final_end, "end": end, "text": text, "speaker": None, "marker": marker}
        self._last_final_end = end
        self.state.sentences.append(sent)
        idx = len(self.state.sentences) - 1
        self._writer.write_segments([Segment(text, sent["start"], sent["end"])])
        self.cb.on_final(sent)
        if self.live_speaker is not None and not marker:
            threading.Thread(target=self._identify, args=(idx, sent["start"], sent["end"]), daemon=True).start()

    def _identify(self, idx: int, start: float, end: float) -> None:
        try:
            pcm = b"".join(self._audio)
            a, b = int(start * SAMPLE_RATE) * 2, int(end * SAMPLE_RATE) * 2
            clip = np.frombuffer(pcm[a:b], dtype="<i2").astype(np.float32) / 32768.0
            if len(clip) < SAMPLE_RATE * 0.8:
                return
            name = self.live_speaker(clip) or "未知"
            self.state.sentences[idx]["speaker"] = name
            self.cb.on_speaker(idx, name)
        except Exception as e:  # noqa: BLE001
            log.debug("实时说话人识别失败：%s", e)

    def _on_client_error(self, err: RealtimeError) -> None:
        if err.kind == "connect":
            self._connected = False
            self.cb.on_status("连接中断")
            if not self.s.auto_reconnect:
                self._fatal = err
            return
        # server-side error: stop the session, keep audio + text
        self._fatal = err
        self.state.error = str(err)
        self.cb.on_error(f"服务端错误：{err}")
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon(self._queue.put_nowait, None)

    def _finish_files(self) -> None:
        try:
            if self._wav is not None:
                self._wav.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._writer.finalize()
        except Exception:  # noqa: BLE001
            pass

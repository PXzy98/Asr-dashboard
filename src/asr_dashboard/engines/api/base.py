"""Shared behaviour of every API engine: limits, compression, splitting, retries, concurrency, cancel."""

from __future__ import annotations

import email.utils
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from asr_dashboard.config.schema import ApiModelConfig
from asr_dashboard.core import audio as audiomod
from asr_dashboard.core import segmentation as segm
from asr_dashboard.core import vad as vadmod
from asr_dashboard.engines.base import BaseASREngine, EngineCapabilities, ProgressFn
from asr_dashboard.models import AudioChunk, CancelledError, CancelToken, Segment, Word
from asr_dashboard.redact import redact, register_secret
from asr_dashboard.settings_fields import SettingField

log = logging.getLogger(__name__)

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
SPLIT_WITHOUT_TIMESTAMPS = 90.0  # a timestamp-less answer for longer audio is re-requested in ~60 s pieces


class APIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, kind: str = "error", body: str = "") -> None:
        super().__init__(redact(message))
        self.status = status
        self.kind = kind  # network / auth / not_found / bad_request / rate_limit / server / timeout / parse
        self.body = redact(body[:2000])


def classify_status(status: int) -> str:
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limit"
    if status >= 500:
        return "server"
    return "bad_request"


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - time.time())
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Upload:
    data: bytes
    filename: str
    mime: str
    offset: float
    duration: float
    samples: np.ndarray


@dataclass
class CallResult:
    raw: Any
    segments: list[Segment]  # times relative to the uploaded audio
    has_timestamps: bool
    language: str | None = None


def common_fields() -> list[SettingField]:
    return [
        SettingField("name", "名称", "str", "API 模型"),
        SettingField("base_url", "API 地址（base_url）", "str", ""),
        SettingField("api_key", "API Key", "secret", "", "保存在系统钥匙串（不可用时为明文文件）"),
        SettingField("model", "模型名", "str", ""),
        SettingField("language", "默认语言", "str", "auto", "auto 或 ISO 代码（zh、en…）"),
        SettingField("supports_context", "支持上下文提示", "bool", True),
        SettingField("timeout", "超时（秒）", "float", 120, minimum=1, maximum=7200),
        SettingField("retries", "重试次数", "int", 3, minimum=0, maximum=20),
        SettingField("concurrency", "并发数", "int", 2, minimum=1, maximum=32),
        SettingField("max_file_mb", "单文件大小上限（MB）", "float", 0, "0 表示不限", minimum=0),
        SettingField("max_audio_seconds", "单次音频时长上限（秒）", "float", 0, "0 表示不限", minimum=0),
        SettingField("timestamps", "请求时间戳", "bool", True),
        SettingField("extra", "额外参数（JSON）", "json", {}, "原样合并进请求"),
        SettingField("options.compress_formats", "超限时可用的压缩格式（JSON 数组）", "json", ["flac", "mp3"],
                     "服务端不支持 FLAC 时改为 [\"mp3\"]"),
    ]


class BaseAPIEngine(BaseASREngine):
    provider_type = ""
    provider_label = ""

    def __init__(self, cfg: ApiModelConfig, api_key: str = "", temp_dir: str | None = None,
                 transport: httpx.BaseTransport | None = None, sleep=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.api_key = api_key or ""
        register_secret(self.api_key)
        self.temp_dir = temp_dir
        self.transport = transport
        self.id = cfg.id
        self.display_name = cfg.name
        self._clients: set[httpx.Client] = set()
        self._clients_lock = threading.Lock()
        self._timestamps_supported = cfg.timestamps
        self._sleep = sleep  # injectable for tests

    # ------------------------------------------------------------------ interface
    @classmethod
    def settings_schema(cls) -> list[SettingField]:
        return common_fields() + cls.provider_fields()

    @classmethod
    def provider_fields(cls) -> list[SettingField]:
        return []

    @property
    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        ts = self._timestamps_supported and self.provider_timestamps()
        return EngineCapabilities(
            word_timestamps=False, segment_timestamps=ts, language_detect=True, languages=[],
            context_prompt=self.cfg.supports_context, max_audio_seconds=self.cfg.max_audio_seconds,
            max_file_mb=self.cfg.max_file_mb, supports_batch=self.cfg.concurrency > 1, is_local=False,
        )

    def provider_timestamps(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def unload(self) -> None:
        self.abort()

    def describe_settings(self) -> dict:
        d = self.cfg.model_dump(mode="json")
        d["provider"] = self.provider_type
        return d

    def wants_chunked_input(self) -> bool:
        # with timestamps we can send the whole file (split only when limits require it)
        return not self.capabilities.segment_timestamps

    def max_chunk_seconds(self) -> float | None:
        return self.cfg.max_audio_seconds or None

    @property
    def batch_size(self) -> int:
        return max(1, self.cfg.concurrency) * 2

    # ------------------------------------------------------------------ HTTP helpers
    def client(self) -> httpx.Client:
        c = httpx.Client(timeout=httpx.Timeout(self.cfg.timeout, connect=min(30.0, self.cfg.timeout)),
                         transport=self.transport, follow_redirects=True)
        with self._clients_lock:
            self._clients.add(c)
        return c

    def release(self, c: httpx.Client) -> None:
        with self._clients_lock:
            self._clients.discard(c)
        c.close()

    def abort(self) -> None:
        """Close every open client -> in-flight requests fail immediately."""
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
        for c in clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass

    def _wait(self, seconds: float, cancel: CancelToken) -> None:
        if self._sleep:
            self._sleep(seconds)
            cancel.raise_if_cancelled()
            return
        if cancel.wait(seconds):
            raise CancelledError("任务已取消")

    def request(self, method: str, url: str, cancel: CancelToken, **kw) -> httpx.Response:
        """HTTP with exponential back-off on 429 / 5xx / timeouts, honouring Retry-After."""
        attempts = self.cfg.retries + 1
        last: Exception | None = None
        for attempt in range(attempts):
            cancel.raise_if_cancelled()
            c = self.client()
            try:
                resp = c.request(method, url, **kw)
                resp.read()
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if cancel.cancelled:
                    raise CancelledError("任务已取消") from e
                kind = "timeout" if isinstance(e, httpx.TimeoutException) else "network"
                last = APIError(f"{'请求超时' if kind == 'timeout' else '网络错误'}：{e}", None, kind)
                if attempt + 1 < attempts:
                    self._wait(min(60.0, 2.0 ** attempt), cancel)
                    continue
                raise last from e
            finally:
                self.release(c)
            if resp.status_code < 400:
                return resp
            body = resp.text
            err = APIError(f"HTTP {resp.status_code}：{body[:500]}", resp.status_code,
                           classify_status(resp.status_code), body)
            if resp.status_code in RETRY_STATUS and attempt + 1 < attempts:
                delay = parse_retry_after(resp.headers.get("retry-after"))
                self._wait(min(120.0, delay if delay is not None else 2.0 ** attempt), cancel)
                last = err
                continue
            raise err
        raise last or APIError("请求失败")

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    # ------------------------------------------------------------------ audio preparation
    def _encode(self, samples: np.ndarray, fmt: str) -> bytes:
        if fmt == "wav":
            buf = io.BytesIO()
            import soundfile as sf

            sf.write(buf, samples, 16000, format="WAV", subtype="PCM_16")
            return buf.getvalue()
        tmp = audiomod.save_wav(audiomod.temp_root(self.temp_dir) / f"api_{id(samples)}_{time.time_ns()}.wav", samples)
        try:
            out = audiomod.encode(tmp, fmt, self.temp_dir)
            data = out.read_bytes()
            audiomod.safe_unlink(out)
            return data
        finally:
            audiomod.safe_unlink(tmp)

    def prepare_uploads(self, chunk: AudioChunk) -> list[Upload]:
        """Split / compress a chunk so every upload respects max_file_mb and max_audio_seconds."""
        samples = chunk.waveform()
        pieces = [(chunk.offset, samples)]
        max_s = self.cfg.max_audio_seconds
        if max_s and len(samples) / 16000 > max_s:
            params = segm.SegParams(strategy="hybrid", target_duration=max_s * 0.9, max_segment=max_s,
                                    min_segment=min(5.0, max_s / 2), overlap=0.5)
            bounds = segm.plan_chunks(samples, params, use_silero=False)
            pieces = [(chunk.offset + b.start, audiomod.slice_samples(samples, b.start, b.end)) for b in bounds]
        out: list[Upload] = []
        for off, s in pieces:
            out.extend(self._fit_size(off, s))
        return out

    def _fit_size(self, offset: float, samples: np.ndarray) -> list[Upload]:
        limit = self.cfg.max_file_mb * 1024 * 1024
        data = self._encode(samples, "wav")
        dur = len(samples) / 16000
        if not limit or len(data) <= limit:
            return [Upload(data, "audio.wav", "audio/wav", offset, dur, samples)]
        mimes = {"flac": "audio/flac", "mp3": "audio/mpeg"}
        allowed = self.cfg.options.get("compress_formats") or ["flac", "mp3"]
        for fmt, mime in ((f, mimes[f]) for f in allowed if f in mimes):
            try:
                enc = self._encode(samples, fmt)
            except Exception as e:  # noqa: BLE001
                log.warning("压缩为 %s 失败：%s", fmt, e)
                continue
            if len(enc) <= limit:
                return [Upload(enc, f"audio.{fmt}", mime, offset, dur, samples)]
        # still too large: split in two at the longest silence near the middle and recurse
        half = dur / 2
        params = segm.SegParams(strategy="hybrid", target_duration=half, max_segment=half * 1.2,
                                min_segment=min(5.0, half / 2), overlap=0.0, search_window=half * 0.2)
        bounds = segm.split_hybrid(dur, vadmod.silences(samples), params)
        if len(bounds) < 2:
            bounds = [segm.Bound(0, half), segm.Bound(half, dur)]
        out = []
        for b in bounds:
            out.extend(self._fit_size(offset + b.start, audiomod.slice_samples(samples, b.start, b.end)))
        return out

    # ------------------------------------------------------------------ transcription
    def call(self, up: Upload, language: str | None, context: str | None, cancel: CancelToken) -> CallResult:
        raise NotImplementedError

    def transcribe(self, chunks: list[AudioChunk], language: str | None, context: str | None,
                   on_progress: ProgressFn, cancel: CancelToken) -> list[Segment]:
        language = language or (None if self.cfg.language in ("", "auto") else self.cfg.language)
        if not self.cfg.supports_context:
            context = None
        uploads = [u for c in chunks for u in self.prepare_uploads(c)]
        total = len(uploads)
        results: list[list[Segment]] = [[] for _ in uploads]
        cancel.on_cancel(self.abort)
        done = 0
        try:
            with ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as ex:
                futs = {ex.submit(self._one, u, language, context, cancel): i for i, u in enumerate(uploads)}
                for f in as_completed(futs):
                    i = futs[f]
                    results[i] = f.result()
                    done += 1
                    on_progress(done, total, f"识别 {done}/{total}")
        except CancelledError:
            raise
        except Exception:
            if cancel.cancelled:
                raise CancelledError("任务已取消") from None
            raise
        finally:
            cancel.remove_callback(self.abort)
        out = [s for r in results for s in r]
        out.sort(key=lambda s: s.start)
        return out

    def _one(self, up: Upload, language, context, cancel: CancelToken) -> list[Segment]:
        cancel.raise_if_cancelled()
        res = self.call(up, language, context, cancel)
        if not res.has_timestamps:
            if up.duration > SPLIT_WITHOUT_TIMESTAMPS:
                # server gave text only for long audio -> re-request in pieces to get usable times
                self._timestamps_supported = False
                params = segm.SegParams(strategy="hybrid", target_duration=60, max_segment=60, min_segment=5)
                bounds = segm.plan_chunks(up.samples, params, use_silero=False)
                out: list[Segment] = []
                for b in bounds:
                    sub = Upload(b"", "", "", up.offset + b.start, b.end - b.start,
                                 audiomod.slice_samples(up.samples, b.start, b.end))
                    for piece in self._fit_size(sub.offset, sub.samples):
                        r = self.call(piece, language, context, cancel)
                        text = " ".join(s.text for s in r.segments).strip()
                        if text:
                            out.append(Segment(text, piece.offset, piece.offset + piece.duration, language=r.language))
                return out
            text = "".join(s.text for s in res.segments).strip() if _cjk(res.segments) else \
                " ".join(s.text.strip() for s in res.segments).strip()
            return [Segment(text, up.offset, up.offset + up.duration, language=res.language)] if text else []
        out = []
        for s in res.segments:
            s.start = min(max(0.0, s.start), up.duration) + up.offset
            s.end = min(max(s.end, 0.0), up.duration) + up.offset
            if s.end <= s.start:
                s.end = min(up.offset + up.duration, s.start + 0.3)
            for w in s.words:
                w.start += up.offset
                w.end += up.offset
            s.language = s.language or res.language
            out.append(s)
        return out

    # ------------------------------------------------------------------ connection test
    def test_connection(self, cancel: CancelToken | None = None) -> dict:
        cancel = cancel or CancelToken()
        clip = audiomod.bundled_test_clip(3.0)
        up = self._fit_size(0.0, clip)[0]
        t0 = time.perf_counter()
        res = self.call(up, None if self.cfg.language in ("", "auto") else self.cfg.language, None, cancel)
        return {"elapsed": time.perf_counter() - t0, "raw": res.raw,
                "segments": [{"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
                             for s in res.segments], "has_timestamps": res.has_timestamps}


def _cjk(segs: list[Segment]) -> bool:
    from asr_dashboard.text.rules import is_cjk_text

    return any(is_cjk_text(s.text) for s in segs)


def words_into_segments(segments: list[Segment], words: list[Word]) -> None:
    for w in words:
        mid = (w.start + w.end) / 2
        for s in segments:
            if s.start - 0.05 <= mid <= s.end + 0.05:
                s.words.append(w)
                break

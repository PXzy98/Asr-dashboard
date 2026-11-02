"""Save-time validation of the realtime service: connect, stream the built-in test clip, expect text in 10 s."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from asr_dashboard.core.audio import bundled_test_clip, float_to_pcm16
from asr_dashboard.realtime.protocols import make_client
from asr_dashboard.realtime.protocols.base import KIND_HINT, RealtimeError, RealtimeEvents


@dataclass
class RealtimeValidation:
    ok: bool
    kind: str = ""
    message: str = ""
    first_text_latency: float | None = None
    total: float = 0.0
    text: str = ""

    @property
    def hint(self) -> str:
        return KIND_HINT.get(self.kind, "")


async def validate_async(protocol: str, url: str, api_key: str, model: str, language: str | None = None,
                         chunk_ms: int = 100, timeout: float = 10.0, connect=None, realtime_pace: bool = False) -> RealtimeValidation:
    if not url.strip():
        return RealtimeValidation(False, "connect", "请填写服务地址")
    client = make_client(protocol, url, api_key, model, language, "", connect)
    finals: list[str] = []
    partials: list[str] = []
    errors: list[RealtimeError] = []
    t0 = time.perf_counter()
    first: list[float] = []
    sent_end: list[float] = []
    ev = RealtimeEvents(on_partial=lambda t: partials.append(t), on_final=finals.append, on_error=errors.append,
                        on_first_text=lambda: first.append(time.perf_counter()))
    try:
        await client.connect(ev)
        clip = bundled_test_clip(0)
        step = int(16000 * chunk_ms / 1000)
        for i in range(0, len(clip), step):
            await client.send_audio(float_to_pcm16(clip[i:i + step]))
            if realtime_pace:
                await asyncio.sleep(chunk_ms / 1000)
        sent_end.append(time.perf_counter())
        await client.finish(timeout)
        if errors:
            raise errors[0]
        text = " ".join(finals).strip() or (partials[-1] if partials else "")
        total = time.perf_counter() - t0
        if not text:
            return RealtimeValidation(False, "timeout", "服务端返回了空的识别结果", None, total)
        lat = (first[0] - t0) if first else None
        return RealtimeValidation(True, "", "可用", lat, total, text)
    except RealtimeError as e:
        return RealtimeValidation(False, e.kind, str(e), None, time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001
        return RealtimeValidation(False, "connect", f"{type(e).__name__}: {e}", None, time.perf_counter() - t0)
    finally:
        await client.close()


def validate(protocol: str, url: str, api_key: str, model: str, language: str | None = None, chunk_ms: int = 100,
             timeout: float = 10.0) -> RealtimeValidation:
    return asyncio.run(validate_async(protocol, url, api_key, model, language, chunk_ms, timeout))

"""Audio capture: microphone (sounddevice) and Windows system audio via WASAPI loopback (PyAudioWPatch).

PyAudioWPatch was chosen over ``soundcard`` for loopback: it ships pre-built Windows wheels, is a thin
fork of PyAudio (PortAudio with WASAPI loopback patches) and is actively released; ``soundcard`` has had
numpy-2 incompatibilities and its loopback depends on per-device COM handling.
Every source is converted to 16 kHz mono float32 frames.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from asr_dashboard.core.audio import SAMPLE_RATE, resample

log = logging.getLogger(__name__)

FrameCb = Callable[[np.ndarray], None]


@dataclass
class Device:
    id: str
    name: str
    kind: str  # microphone / system
    default_rate: int = 48000
    channels: int = 1


def loopback_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import pyaudiowpatch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def list_microphones() -> list[Device]:
    try:
        import sounddevice as sd

        out = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append(Device(str(i), d["name"], "microphone", int(d.get("default_samplerate") or 48000),
                                  min(2, int(d["max_input_channels"]))))
        return out
    except Exception as e:  # noqa: BLE001
        log.info("无法枚举麦克风：%s", e)
        return []


def list_loopback() -> list[Device]:
    if not loopback_available():
        return []
    import pyaudiowpatch as pa

    p = pa.PyAudio()
    try:
        out = []
        for d in p.get_loopback_device_info_generator():
            out.append(Device(str(d["index"]), d["name"], "system", int(d["defaultSampleRate"]),
                              int(d["maxInputChannels"]) or 2))
        return out
    finally:
        p.terminate()


class _Source:
    def __init__(self, dev: Device | None, cb: FrameCb) -> None:
        self.dev = dev
        self.cb = cb

    def _emit(self, data: np.ndarray, rate: int) -> None:
        if data.ndim > 1:
            data = data.mean(axis=1)
        self.cb(resample(data.astype(np.float32), rate, SAMPLE_RATE))


class MicSource(_Source):
    def start(self) -> None:
        import sounddevice as sd

        rate = SAMPLE_RATE
        dev = int(self.dev.id) if self.dev else None
        try:
            sd.check_input_settings(device=dev, samplerate=SAMPLE_RATE, channels=1)
        except Exception:  # noqa: BLE001
            rate = self.dev.default_rate if self.dev else 48000
        self.rate = rate
        self.stream = sd.InputStream(device=dev, samplerate=rate, channels=1, dtype="float32",
                                     blocksize=int(rate * 0.05),
                                     callback=lambda indata, frames, t, status: self._emit(indata.copy(), rate))
        self.stream.start()

    def stop(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:  # noqa: BLE001
            pass


class LoopbackSource(_Source):
    def start(self) -> None:
        import pyaudiowpatch as pa

        self.p = pa.PyAudio()
        if self.dev is None:
            info = self.p.get_default_wasapi_loopback()
            self.dev = Device(str(info["index"]), info["name"], "system", int(info["defaultSampleRate"]),
                              int(info["maxInputChannels"]) or 2)
        rate, ch = self.dev.default_rate, self.dev.channels

        def callback(in_data, frame_count, time_info, status):
            data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, ch)
            self._emit(data, rate)
            return (None, pa.paContinue)

        self.stream = self.p.open(format=pa.paFloat32, channels=ch, rate=rate, input=True,
                                  input_device_index=int(self.dev.id), frames_per_buffer=int(rate * 0.05),
                                  stream_callback=callback)
        self.stream.start_stream()

    def stop(self) -> None:
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except Exception:  # noqa: BLE001
            pass


class Mixer:
    """Mixes two 16 kHz streams sample-by-sample (whichever arrives first waits briefly for the other)."""

    def __init__(self, cb: FrameCb) -> None:
        self.cb = cb
        self.buf = {0: np.zeros(0, np.float32), 1: np.zeros(0, np.float32)}
        self.lock = threading.Lock()

    def push(self, k: int, frames: np.ndarray) -> None:
        with self.lock:
            self.buf[k] = np.concatenate([self.buf[k], frames])
            n = min(len(self.buf[0]), len(self.buf[1]))
            # if one side is silent/stalled for > 0.5 s, pass the other through
            if n == 0 and max(len(self.buf[0]), len(self.buf[1])) > SAMPLE_RATE // 2:
                k2 = 0 if len(self.buf[0]) else 1
                out, self.buf[k2] = self.buf[k2], np.zeros(0, np.float32)
                self.cb(out)
                return
            if n:
                out = np.clip(self.buf[0][:n] + self.buf[1][:n], -1, 1)
                self.buf[0], self.buf[1] = self.buf[0][n:], self.buf[1][n:]
                self.cb(out)


class AudioCapture:
    def __init__(self, source: str, cb: FrameCb, mic: Device | None = None, system: Device | None = None) -> None:
        self.source = source
        self.cb = cb
        self.mic = mic
        self.system = system
        self.sources: list[_Source] = []

    def start(self) -> None:
        if self.source in ("system", "mixed") and not loopback_available():
            raise RuntimeError("系统声音采集需要 Windows + PyAudioWPatch（WASAPI 回环）")
        if self.source == "microphone":
            self.sources = [MicSource(self.mic, self.cb)]
        elif self.source == "system":
            self.sources = [LoopbackSource(self.system, self.cb)]
        else:
            mix = Mixer(self.cb)
            self.sources = [MicSource(self.mic, lambda f: mix.push(0, f)),
                            LoopbackSource(self.system, lambda f: mix.push(1, f))]
        for s in self.sources:
            s.start()

    def stop(self) -> None:
        for s in self.sources:
            s.stop()
        self.sources = []

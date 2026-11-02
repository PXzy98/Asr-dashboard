"""Voice activity detection: Silero VAD (silero-vad package) with an energy-based fallback."""

from __future__ import annotations

import logging
import threading

import numpy as np

from asr_dashboard.core.audio import SAMPLE_RATE

log = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def silero_available() -> bool:
    try:
        import silero_vad  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _load():
    global _model
    with _lock:
        if _model is None:
            from silero_vad import load_silero_vad

            _model = load_silero_vad()
        return _model


def speech_regions(samples: np.ndarray, sr: int = SAMPLE_RATE, threshold: float = 0.5,
                   min_silence: float = 0.3, min_speech: float = 0.25,
                   use_silero: bool = True) -> list[tuple[float, float]]:
    """Return speech regions in seconds."""
    if len(samples) == 0:
        return []
    if use_silero and silero_available():
        try:
            import torch
            from silero_vad import get_speech_timestamps

            model = _load()
            ts = get_speech_timestamps(
                torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)), model,
                sampling_rate=sr, threshold=threshold,
                min_silence_duration_ms=int(min_silence * 1000),
                min_speech_duration_ms=int(min_speech * 1000), return_seconds=True,
            )
            return [(float(t["start"]), float(t["end"])) for t in ts]
        except Exception as e:  # noqa: BLE001
            log.warning("Silero VAD 失败，改用能量阈值检测：%s", e)
    return energy_regions(samples, sr, min_silence=min_silence, min_speech=min_speech)


def frame_db(samples: np.ndarray, sr: int = SAMPLE_RATE, frame: float = 0.03) -> np.ndarray:
    n = max(1, int(frame * sr))
    usable = len(samples) // n * n
    if usable == 0:
        return np.array([-120.0])
    frames = samples[:usable].reshape(-1, n).astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


def energy_regions(samples: np.ndarray, sr: int = SAMPLE_RATE, threshold_db: float = -40.0,
                   min_silence: float = 0.3, min_speech: float = 0.25,
                   frame: float = 0.03) -> list[tuple[float, float]]:
    db = frame_db(samples, sr, frame)
    voiced = db > threshold_db
    regions: list[tuple[float, float]] = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append((start * frame, i * frame))
            start = None
    if start is not None:
        regions.append((start * frame, len(voiced) * frame))
    # merge gaps shorter than min_silence, drop short blips
    merged: list[tuple[float, float]] = []
    for a, b in regions:
        if merged and a - merged[-1][1] < min_silence:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_speech]


def silences(samples: np.ndarray, sr: int = SAMPLE_RATE, threshold_db: float = -40.0,
             min_silence: float = 0.3, frame: float = 0.03) -> list[tuple[float, float]]:
    """Silent stretches (seconds) at least ``min_silence`` long, by energy threshold."""
    db = frame_db(samples, sr, frame)
    quiet = db <= threshold_db
    out: list[tuple[float, float]] = []
    start = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * frame >= min_silence:
                out.append((start * frame, i * frame))
            start = None
    if start is not None and (len(quiet) - start) * frame >= min_silence:
        out.append((start * frame, len(quiet) * frame))
    return out

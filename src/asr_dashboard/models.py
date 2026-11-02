"""Core data model shared by engines, pipeline, export and UI."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None
    confidence: float | None = None


@dataclass
class Segment:
    text: str
    start: float
    end: float
    speaker: str | None = None
    language: str | None = None
    words: list[Word] = field(default_factory=list)
    original_text: str | None = None  # text before post-processing (diff / undo)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SpeakerInfo:
    label: str
    display_name: str | None = None
    voiceprint_id: str | None = None
    match_score: float | None = None
    embedding: list[float] | None = None

    @property
    def name(self) -> str:
        return self.display_name or self.label


@dataclass
class TranscriptionResult:
    source_file: str
    segments: list[Segment]
    text: str
    language: str | None
    duration: float
    speakers: dict[str, SpeakerInfo] = field(default_factory=dict)
    overlaps: list[tuple[float, float, list[str]]] = field(default_factory=list)
    engine_id: str = ""
    engine_settings: dict = field(default_factory=dict)
    context_used: str | None = None
    postprocess_mode: str = "rules"
    summary: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    speaker_suggestions: list[dict] = field(default_factory=list)  # LLM speaker guesses (need confirmation)
    draft_text: str | None = None  # realtime draft, kept for comparison after refinement

    # ---- helpers -------------------------------------------------------
    def speaker_name(self, label: str | None) -> str | None:
        if label is None:
            return None
        info = self.speakers.get(label)
        return info.name if info else label

    def rebuild_text(self) -> None:
        self.text = "\n".join(s.text for s in self.segments if s.text)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overlaps"] = [[a, b, list(c)] for a, b, c in self.overlaps]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TranscriptionResult:
        segs = []
        for s in d.get("segments", []):
            words = [Word(**w) for w in s.get("words", [])]
            segs.append(
                Segment(
                    text=s["text"], start=s["start"], end=s["end"], speaker=s.get("speaker"),
                    language=s.get("language"), words=words, original_text=s.get("original_text"),
                )
            )
        speakers = {k: SpeakerInfo(**v) for k, v in (d.get("speakers") or {}).items()}
        return cls(
            source_file=d.get("source_file", ""), segments=segs, text=d.get("text", ""),
            language=d.get("language"), duration=d.get("duration", 0.0), speakers=speakers,
            overlaps=[(o[0], o[1], list(o[2])) for o in d.get("overlaps", [])],
            engine_id=d.get("engine_id", ""), engine_settings=d.get("engine_settings", {}),
            context_used=d.get("context_used"), postprocess_mode=d.get("postprocess_mode", "rules"),
            summary=d.get("summary"), timings=d.get("timings", {}), warnings=d.get("warnings", []),
            speaker_suggestions=d.get("speaker_suggestions", []), draft_text=d.get("draft_text"),
        )


@dataclass
class AudioChunk:
    """A piece of the (16 kHz mono) source audio. ``offset`` is its start time in the original file."""

    offset: float
    duration: float
    path: Path | None = None
    samples: np.ndarray | None = None
    sample_rate: int = 16000
    index: int = 0

    @property
    def end(self) -> float:
        return self.offset + self.duration

    def waveform(self) -> np.ndarray:
        if self.samples is not None:
            return self.samples
        if self.path is None:
            raise ValueError("AudioChunk has neither samples nor path")
        import soundfile as sf

        data, _sr = sf.read(str(self.path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data


class CancelledError(Exception):
    """Raised when a task is cancelled by the user."""


class CancelToken:
    """Thread-safe cancellation flag with callbacks (e.g. to kill subprocesses / abort requests)."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._callbacks: list = []
        self._lock = threading.Lock()

    def cancel(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb()
            except Exception:  # noqa: BLE001 - callbacks must never break cancellation
                pass

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("任务已取消")

    def on_cancel(self, cb) -> None:
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(cb)
                return
        cb()

    def remove_callback(self, cb) -> None:
        with self._lock:
            if cb in self._callbacks:
                self._callbacks.remove(cb)

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)

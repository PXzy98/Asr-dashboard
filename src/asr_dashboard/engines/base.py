"""Engine interface shared by the local Qwen3 engine and every API provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from asr_dashboard.models import AudioChunk, CancelToken, Segment
from asr_dashboard.settings_fields import SettingField

ProgressFn = Callable[[int, int, str], None]


@dataclass
class EngineCapabilities:
    word_timestamps: bool = False
    segment_timestamps: bool = False
    language_detect: bool = True
    languages: list[str] = field(default_factory=list)  # ISO codes; empty = unknown/any
    context_prompt: bool = False
    max_audio_seconds: float = 0.0  # 0 = unlimited
    max_file_mb: float = 0.0
    supports_batch: bool = False
    is_local: bool = False
    # local-only knobs the pipeline needs to know about
    forced_alignment: bool = False  # engine can produce word timestamps itself when enabled


class BaseASREngine(ABC):
    id: str = ""
    display_name: str = ""
    capabilities: EngineCapabilities = EngineCapabilities()

    # pipeline reads these after transcribe() to surface non-fatal problems
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.info_messages: list[str] = []  # shown as InfoBar (e.g. OOM -> CPU fallback)

    @classmethod
    @abstractmethod
    def settings_schema(cls) -> list[SettingField]: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def transcribe(
        self,
        chunks: list[AudioChunk],
        language: str | None,
        context: str | None,
        on_progress: ProgressFn,
        cancel: CancelToken,
    ) -> list[Segment]:
        """Return segments whose times are absolute (chunk offsets already applied)."""

    # optional hooks ---------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return True

    def describe_settings(self) -> dict:
        """Settings snapshot written to JSON export (must not contain secrets)."""
        return {}

    def max_chunk_seconds(self) -> float | None:
        """Upper bound for segmentation chunk length imposed by the engine (e.g. aligner 300s)."""
        return None

    def wants_chunked_input(self) -> bool:
        """Whether the pipeline should split audio before calling transcribe()."""
        return True

    def reduce_batch_size(self) -> bool:
        """Halve inference batch size after OOM; returns False if not supported / already 1."""
        return False

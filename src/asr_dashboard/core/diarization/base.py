"""Diarization data types and backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from asr_dashboard.models import CancelToken


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class DiarizationOutput:
    turns: list[SpeakerTurn]  # may overlap (used for overlap detection)
    exclusive_turns: list[SpeakerTurn]  # one speaker per instant (used to assign text)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # backend-native centroids (optional)
    backend: str = ""

    @property
    def speakers(self) -> list[str]:
        return sorted({t.speaker for t in self.exclusive_turns or self.turns})


@dataclass
class DiarizeRequest:
    wav_path: str
    samples: np.ndarray | None
    num_speakers: int = 0
    min_speakers: int = 1
    max_speakers: int = 20
    device: str = "cpu"


ProgressCb = Callable[[float, str], None]


class BaseDiarizer(ABC):
    name = ""
    display_name = ""

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def diarize(self, req: DiarizeRequest, progress: ProgressCb, cancel: CancelToken) -> DiarizationOutput: ...

    def unload(self) -> None:  # noqa: B027 - optional
        pass


def normalize_labels(out: DiarizationOutput) -> DiarizationOutput:
    """Rename speakers to SPEAKER_00.. in order of first appearance."""
    order: list[str] = []
    for t in sorted(out.exclusive_turns or out.turns, key=lambda t: t.start):
        if t.speaker not in order:
            order.append(t.speaker)
    for t in out.turns:
        if t.speaker not in order:
            order.append(t.speaker)
    mapping = {old: f"SPEAKER_{i:02d}" for i, old in enumerate(order)}
    return DiarizationOutput(
        turns=[SpeakerTurn(t.start, t.end, mapping[t.speaker]) for t in out.turns],
        exclusive_turns=[SpeakerTurn(t.start, t.end, mapping[t.speaker]) for t in out.exclusive_turns],
        embeddings={mapping[k]: v for k, v in out.embeddings.items() if k in mapping},
        backend=out.backend,
    )

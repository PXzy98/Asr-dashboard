"""Unified voiceprint embedding model (all library vectors must come from the same model)."""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)

MODEL_IDS = {"pyannote": "pyannote/speaker-diarization-community-1#embedding",
             "wavlm": "microsoft/wavlm-base-plus-sv"}


class VoiceprintEmbedder:
    """Embeds a clip; long clips are cut into 3 s windows and averaged."""

    def __init__(self, kind: str, device: str = "cpu", pyannote_dir: str | None = None,
                 wavlm_model: str = "microsoft/wavlm-base-plus-sv", pyannote_diarizer=None) -> None:
        self.kind = kind
        self.device = device
        self.pyannote_dir = pyannote_dir
        self.wavlm_model = wavlm_model
        self._pyannote = pyannote_diarizer
        self._wavlm = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return MODEL_IDS["pyannote"] if self.kind == "pyannote" else self.wavlm_model

    def available(self) -> tuple[bool, str]:
        if self.kind == "pyannote":
            from asr_dashboard.core.diarization import pyannote_backend

            ok, reason = pyannote_backend.availability(self.pyannote_dir)
            return ok, reason
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"WavLM 需要 torch 与 transformers：{e}"

    def _raw(self, clip: np.ndarray) -> np.ndarray:
        if self.kind == "pyannote":
            if self._pyannote is None:
                from asr_dashboard.core.diarization.pyannote_backend import PyannoteDiarizer

                self._pyannote = PyannoteDiarizer(self.pyannote_dir, self.device)
            return self._pyannote.embed(clip)
        if self._wavlm is None:
            from asr_dashboard.core.diarization.embedding_backend import WavLMEmbedder

            self._wavlm = WavLMEmbedder(self.wavlm_model, self.device)
        return self._wavlm([clip])[0]

    def embed(self, samples: np.ndarray, window: float = 3.0, max_windows: int = 20) -> np.ndarray:
        sr = 16000
        n = int(window * sr)
        with self._lock:
            if len(samples) <= n * 1.5:
                vecs = [self._raw(samples)]
            else:
                starts = list(range(0, len(samples) - n + 1, n))[:max_windows]
                vecs = [self._raw(samples[s:s + n]) for s in starts]
        vecs = [v / max(np.linalg.norm(v), 1e-9) for v in vecs if np.all(np.isfinite(v))]
        if not vecs:
            raise RuntimeError("无法提取声纹（音频过短或静音）")
        m = np.mean(vecs, axis=0)
        return (m / max(np.linalg.norm(m), 1e-9)).astype(np.float32)

    __call__ = embed


def check_sample_quality(samples: np.ndarray, multi_speaker: bool | None = None) -> list[str]:
    """Warnings for library samples (recommended: clean single speaker, >= 10 s)."""
    warns = []
    dur = len(samples) / 16000
    if dur < 10:
        warns.append(f"样本时长 {dur:.1f} 秒，建议至少 10 秒的干净单人语音")
    from asr_dashboard.core.audio import rms_db

    if rms_db(samples) < -45:
        warns.append("样本音量过低，可能大部分是静音")
    if multi_speaker:
        warns.append("检测到样本中可能有多个说话人，建议只保留一个人的声音")
    return warns

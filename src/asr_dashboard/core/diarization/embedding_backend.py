"""Fallback diarization: WavLM x-vectors on sliding windows + clustering (no gated models needed)."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from asr_dashboard.core.diarization.base import (
    BaseDiarizer,
    DiarizationOutput,
    DiarizeRequest,
    ProgressCb,
    SpeakerTurn,
)
from asr_dashboard.models import CancelToken

log = logging.getLogger(__name__)

EmbedFn = Callable[[list[np.ndarray]], np.ndarray]  # batch of clips -> (n, dim)


class WavLMEmbedder:
    """``transformers.WavLMForXVector`` + ``microsoft/wavlm-base-plus-sv``."""

    def __init__(self, model_id: str = "microsoft/wavlm-base-plus-sv", device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self.model = None
        self.fe = None

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.fe = AutoFeatureExtractor.from_pretrained(self.model_id)
        self.model = WavLMForXVector.from_pretrained(self.model_id).to(self.device).eval()

    def __call__(self, clips: list[np.ndarray]) -> np.ndarray:
        import torch

        self.load()
        out = []
        for i in range(0, len(clips), 32):
            batch = [np.asarray(c, dtype=np.float32) for c in clips[i:i + 32]]
            inputs = self.fe(batch, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                emb = self.model(**inputs).embeddings
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out.append(emb.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 512), np.float32)

    def unload(self) -> None:
        self.model = None


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def cluster_embeddings(emb: np.ndarray, num_speakers: int = 0, min_speakers: int = 1, max_speakers: int = 20,
                       method: str = "agglomerative", threshold: float = 0.7) -> np.ndarray:
    """Return a label per row. With num_speakers=0 the count is estimated (cosine-similarity threshold)."""
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    emb = _l2(emb.astype(np.float64))
    max_k = max(1, min(max_speakers, n))
    min_k = max(1, min(min_speakers, max_k))
    if method == "spectral":
        from sklearn.cluster import SpectralClustering

        aff = np.clip(emb @ emb.T, 0, 1)
        k = num_speakers if num_speakers > 0 else _eigengap(aff, min_k, max_k)
        if k <= 1:
            return np.zeros(n, dtype=int)
        return SpectralClustering(n_clusters=k, affinity="precomputed", random_state=0).fit_predict(aff)
    from sklearn.cluster import AgglomerativeClustering

    if num_speakers > 0:
        k = min(num_speakers, n)
        return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(emb)
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1.0 - threshold).fit_predict(emb)
    k = len(set(labels))
    if k < min_k or k > max_k:
        k = min(max(k, min_k), max_k)
        labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(emb)
    return labels


def _eigengap(aff: np.ndarray, min_k: int, max_k: int) -> int:
    d = np.diag(1.0 / np.sqrt(np.maximum(aff.sum(axis=1), 1e-9)))
    lap = np.eye(len(aff)) - d @ aff @ d
    vals = np.sort(np.linalg.eigvalsh(lap))
    gaps = np.diff(vals[: max_k + 1])
    if len(gaps) == 0:
        return 1
    k = int(np.argmax(gaps[min_k - 1:]) + min_k)
    return max(min_k, min(k, max_k))


def windows_to_turns(times: list[tuple[float, float]], labels: np.ndarray) -> list[SpeakerTurn]:
    """Each window votes for its centre region; consecutive same-label windows become one turn."""
    turns: list[SpeakerTurn] = []
    for i, ((a, b), lab) in enumerate(zip(times, labels, strict=True)):
        start = a if i == 0 or times[i - 1][1] <= a else (a + times[i - 1][1]) / 2
        end = b if i + 1 == len(times) or times[i + 1][0] >= b else (b + times[i + 1][0]) / 2
        spk = f"S{int(lab)}"
        if turns and turns[-1].speaker == spk and start - turns[-1].end < 0.05:
            turns[-1].end = end
        else:
            turns.append(SpeakerTurn(start, end, spk))
    return turns


class EmbeddingDiarizer(BaseDiarizer):
    name = "embedding"
    display_name = "WavLM 声纹聚类（兜底）"

    def __init__(self, model_id: str, device: str = "cpu", clustering: str = "agglomerative",
                 threshold: float = 0.7, window: float = 1.5, step: float = 0.75,
                 embed_fn: EmbedFn | None = None, use_silero: bool = True) -> None:
        self.embedder = embed_fn or WavLMEmbedder(model_id, device)
        self.clustering = clustering
        self.threshold = threshold
        self.window = window
        self.step = step
        self.use_silero = use_silero

    def load(self) -> None:
        if hasattr(self.embedder, "load"):
            self.embedder.load()

    def unload(self) -> None:
        if hasattr(self.embedder, "unload"):
            self.embedder.unload()

    def diarize(self, req: DiarizeRequest, progress: ProgressCb, cancel: CancelToken) -> DiarizationOutput:
        from asr_dashboard.core import vad
        from asr_dashboard.core.audio import load_wav

        samples = req.samples if req.samples is not None else load_wav(req.wav_path)
        regions = vad.speech_regions(samples, use_silero=self.use_silero)
        times: list[tuple[float, float]] = []
        for a, b in regions:
            t = a
            while t < b:
                e = min(b, t + self.window)
                if e - t >= min(0.5, self.window):
                    times.append((t, e))
                if e >= b:
                    break
                t += self.step
        if not times:
            return DiarizationOutput([], [], backend=self.name)
        clips = [samples[int(a * 16000):int(b * 16000)] for a, b in times]
        embs = []
        for i in range(0, len(clips), 64):
            cancel.raise_if_cancelled()
            embs.append(self.embedder(clips[i:i + 64]))
            progress(min(0.9, (i + 64) / len(clips) * 0.9), f"提取声纹 {min(i + 64, len(clips))}/{len(clips)}")
        emb = np.concatenate(embs, axis=0)
        labels = cluster_embeddings(emb, req.num_speakers, req.min_speakers, req.max_speakers,
                                    self.clustering, self.threshold)
        turns = windows_to_turns(times, labels)
        cents = {f"S{int(lab)}": _l2(emb[labels == lab].mean(axis=0)).astype(np.float32) for lab in set(labels)}
        progress(1.0, "聚类完成")
        return DiarizationOutput(turns=turns, exclusive_turns=[SpeakerTurn(t.start, t.end, t.speaker) for t in turns],
                                 embeddings=cents, backend=self.name)

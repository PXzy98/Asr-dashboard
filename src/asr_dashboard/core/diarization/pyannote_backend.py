"""pyannote.audio 4.x + ``pyannote/speaker-diarization-community-1`` (CC-BY-4.0).

Verified against pyannote.audio 4.0.7 source:
* ``Pipeline.from_pretrained(checkpoint, revision=None, hparams_file=None, subfolder=None, token=None,
  cache_dir=None)`` — checkpoint may be a local directory containing config.yaml (offline use).
* ``pipeline(file, num_speakers=, min_speakers=, max_speakers=, hook=)`` returns ``DiarizeOutput`` with
  ``speaker_diarization`` (overlap-aware), ``exclusive_speaker_diarization`` (no overlap, meant for
  transcription) and ``speaker_embeddings`` ((num_speakers, dim), in ``speaker_diarization.labels()`` order).
* ``file`` may be ``{"waveform": (channel, time) tensor, "sample_rate": sr}`` which avoids torchcodec decoding.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from asr_dashboard.core.diarization.base import (
    BaseDiarizer,
    DiarizationOutput,
    DiarizeRequest,
    ProgressCb,
    SpeakerTurn,
)
from asr_dashboard.models import CancelledError, CancelToken

log = logging.getLogger(__name__)

REPO_ID = "pyannote/speaker-diarization-community-1"
MODEL_PAGE = "https://huggingface.co/pyannote/speaker-diarization-community-1"
TOKEN_PAGE = "https://huggingface.co/settings/tokens"


def default_model_dir() -> Path:
    from asr_dashboard.paths import models_dir

    return models_dir() / "pyannote-community-1"


def model_ready(model_dir: str | Path | None) -> bool:
    d = Path(model_dir) if model_dir else default_model_dir()
    return (d / "config.yaml").exists()


def pyannote_version() -> str | None:
    try:
        import importlib.metadata as md

        return md.version("pyannote.audio")
    except Exception:  # noqa: BLE001
        return None


def availability(model_dir: str | None) -> tuple[bool, str]:
    v = pyannote_version()
    if not v:
        return False, "未安装 pyannote.audio（uv sync 会安装）"
    if not v.startswith("4."):
        return False, f"需要 pyannote.audio 4.x，当前 {v}"
    if not model_ready(model_dir):
        return False, "尚未下载 Community-1 模型（设置 → 说话人分离 → pyannote 下载引导）"
    return True, ""


def download(token: str, model_dir: str | None = None, progress: ProgressCb | None = None) -> Path:
    """Download the gated pipeline to a local directory (needs a HF token + accepted terms on the model page)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    target = Path(model_dir) if model_dir else default_model_dir()
    target.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.05, "正在下载 pyannote Community-1…")
    try:
        snapshot_download(REPO_ID, local_dir=str(target), token=token or None)
    except GatedRepoError as e:
        raise RuntimeError(f"没有访问权限：请先打开 {MODEL_PAGE} 登录并同意使用条款，然后重试") from e
    except RepositoryNotFoundError as e:
        raise RuntimeError("HuggingFace Token 无效或没有权限访问该模型") from e
    if progress:
        progress(1.0, "下载完成")
    if not model_ready(target):
        raise RuntimeError(f"下载后未找到 {target / 'config.yaml'}")
    return target


class PyannoteDiarizer(BaseDiarizer):
    name = "pyannote"
    display_name = "pyannote Community-1"

    def __init__(self, model_dir: str | None, device: str = "cpu") -> None:
        self.model_dir = str(model_dir or default_model_dir())
        self.device = device
        self.pipeline = None

    def load(self) -> None:
        if self.pipeline is not None:
            return
        import torch
        from pyannote.audio import Pipeline

        if not model_ready(self.model_dir):
            raise RuntimeError("pyannote Community-1 模型尚未下载到本地，请先在设置页完成下载")
        self.pipeline = Pipeline.from_pretrained(self.model_dir)  # local dir -> offline, no token
        self.pipeline.to(torch.device(self.device))
        log.info("pyannote pipeline 已加载：%s（%s）", self.model_dir, self.device)

    def unload(self) -> None:
        self.pipeline = None

    def diarize(self, req: DiarizeRequest, progress: ProgressCb, cancel: CancelToken) -> DiarizationOutput:
        import torch

        self.load()
        samples = req.samples
        if samples is None:
            from asr_dashboard.core.audio import load_wav

            samples = load_wav(req.wav_path)
        file = {"waveform": torch.from_numpy(np.ascontiguousarray(samples))[None, :], "sample_rate": 16000}

        def hook(step_name, step_artifact, file=None, total=None, completed=None):  # noqa: ARG001
            if cancel.cancelled:
                raise CancelledError("任务已取消")
            if total:
                progress(min(1.0, (completed or 0) / total), f"说话人分离：{step_name}")

        kwargs = {}
        if req.num_speakers > 0:
            kwargs["num_speakers"] = req.num_speakers
        else:
            kwargs["min_speakers"] = req.min_speakers
            kwargs["max_speakers"] = req.max_speakers
        out = self.pipeline(file, hook=hook, **kwargs)
        sd = getattr(out, "speaker_diarization", out)
        ex = getattr(out, "exclusive_speaker_diarization", None) or sd
        turns = [SpeakerTurn(s.start, s.end, str(spk)) for s, _, spk in sd.itertracks(yield_label=True)]
        ex_turns = [SpeakerTurn(s.start, s.end, str(spk)) for s, _, spk in ex.itertracks(yield_label=True)]
        emb: dict[str, np.ndarray] = {}
        se = getattr(out, "speaker_embeddings", None)
        if se is not None:
            for label, vec in zip(sd.labels(), se, strict=False):
                if np.all(np.isfinite(vec)):
                    emb[str(label)] = np.asarray(vec, dtype=np.float32)
        return DiarizationOutput(turns=turns, exclusive_turns=ex_turns, embeddings=emb, backend=self.name)

    # used by the voiceprint embedder ("pyannote" model) ---------------------------------
    def embed(self, samples: np.ndarray) -> np.ndarray:
        """Speaker embedding of a mono 16 kHz clip with Community-1's own embedding model."""
        import torch

        self.load()
        model = getattr(self.pipeline, "_embedding", None)
        if model is None:
            raise RuntimeError("当前 pyannote pipeline 没有暴露声纹模型")
        wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))[None, None, :]
        vec = model(wav)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

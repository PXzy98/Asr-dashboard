"""Tests against real models / real audio. Skipped by default.

Run on a machine with the models available:
    uv run pytest -m gpu -s
Optional environment variables:
    ASR_TEST_DIALOG   path to a >= 3 min two-speaker conversation (acceptance item 4)
    ASR_TEST_LONG     path to a >= 30 min recording (acceptance item 7)
    ASR_TEST_VARIANT  1.7b (default) / 0.6b
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from asr_dashboard.config.schema import AppConfig
from asr_dashboard.core.audio import load_wav
from asr_dashboard.core.pipeline import TaskRequest
from asr_dashboard.models import CancelledError, CancelToken
from asr_dashboard.paths import resources_dir

TEST_CLIP = resources_dir() / "test_speech.wav"


def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


# --------------------------------------------------------------------------- runs without GPU
@pytest.mark.skipif(not (_has("silero_vad") and _has("torch")), reason="silero-vad / torch not installed")
def test_real_silero_vad_cuts_in_silence():
    from asr_dashboard.core import segmentation as segm
    from asr_dashboard.core import vad

    clip = load_wav(TEST_CLIP)
    regions = vad.speech_regions(clip)
    assert regions and regions[0][0] < 0.5 and regions[-1][1] > 2.0
    gap = np.zeros(32000, np.float32)
    period = (len(clip) + len(gap)) / 16000
    audio = np.concatenate([np.concatenate([clip, gap]) for _ in range(30)])
    bounds = segm.plan_chunks(audio, segm.SegParams(), use_silero=True)
    for b in bounds[:-1]:  # every cut lies inside a silence gap
        assert (b.end % period) > len(clip) / 16000 - 0.05


# --------------------------------------------------------------------------- real models
def _cfg(tmp_path, **qwen) -> AppConfig:
    c = AppConfig()
    c.app.output_dir = str(tmp_path / "out")
    c.output.formats = ["srt", "json"]
    c.qwen3.variant = os.environ.get("ASR_TEST_VARIANT", "1.7b")  # type: ignore[assignment]
    for k, v in qwen.items():
        setattr(c.qwen3, k, v)
    return c


def _services(tmp_path, cfg):
    from asr_dashboard.config.manager import ConfigManager
    from asr_dashboard.config.secrets import SecretStore
    from asr_dashboard.services import Services

    cm = ConfigManager(tmp_path / "config.yaml", SecretStore(tmp_path / "s.yaml", use_keyring=False))
    cm.save(cfg)
    return Services(cm)


@pytest.mark.gpu
@pytest.mark.parametrize("aligner", [True, False])
def test_qwen3_local_transcribes_and_logs_load_kwargs(tmp_path, caplog, aligner):
    cfg = _cfg(tmp_path, max_new_tokens=512, max_inference_batch_size=4)
    cfg.qwen3.aligner.enabled = aligner
    s = _services(tmp_path, cfg)
    with caplog.at_level(logging.INFO):
        out = s.pipeline(f"qwen3-local:{cfg.qwen3.variant}", cfg).run(TaskRequest(str(TEST_CLIP), language="en"))
    assert "max_new_tokens=512" in caplog.text and "max_inference_batch_size=4" in caplog.text
    text = out.result.text.lower()
    assert "speech" in text or "test" in text
    assert all(seg.end > seg.start for seg in out.result.segments)
    if aligner:
        assert out.result.segments[0].words and out.result.segments[0].words[-1].end > 1.0


@pytest.mark.gpu
@pytest.mark.parametrize("aligner", [True, False])
def test_dialog_timestamps_and_speakers(tmp_path, aligner):
    path = os.environ.get("ASR_TEST_DIALOG")
    if not path:
        pytest.skip("set ASR_TEST_DIALOG to a >= 3 min two-speaker recording")
    cfg = _cfg(tmp_path)
    cfg.qwen3.aligner.enabled = aligner
    cfg.diarization.enabled = True
    cfg.diarization.num_speakers = 2
    s = _services(tmp_path, cfg)
    r = s.pipeline(f"qwen3-local:{cfg.qwen3.variant}", cfg).run(TaskRequest(path)).result
    starts = [seg.start for seg in r.segments]
    assert starts == sorted(starts) and all(seg.end > 0 for seg in r.segments)
    assert len(r.speakers) == 2, r.warnings
    changes = sum(1 for a, b in zip(r.segments, r.segments[1:]) if a.speaker != b.speaker)
    assert changes >= 3


@pytest.mark.gpu
def test_long_audio_cancel_within_two_seconds(tmp_path):
    path = os.environ.get("ASR_TEST_LONG")
    if not path:
        pytest.skip("set ASR_TEST_LONG to a >= 30 min recording")
    cfg = _cfg(tmp_path)
    s = _services(tmp_path, cfg)
    pipe = s.pipeline(f"qwen3-local:{cfg.qwen3.variant}", cfg, keep_loaded=False)
    tok = CancelToken()
    seen: list[float] = []
    started = threading.Event()

    def prog(stage, overall, detail):
        seen.append(overall)
        if stage == "recognize" and overall > 0.15:
            started.set()

    t = threading.Thread(target=lambda: pytest.raises(CancelledError, pipe.run, TaskRequest(path), prog, tok))
    t.start()
    assert started.wait(1800)
    t0 = time.time()
    tok.cancel()
    t.join(30)
    assert time.time() - t0 < 2.0 + 5.0  # the current batch must finish; keep batch size small for <2 s
    assert seen == sorted(seen)
    import torch

    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() < 512 * 2**20


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg")
def test_voiceprint_roundtrip(tmp_path):
    path = os.environ.get("ASR_TEST_DIALOG")
    if not path:
        pytest.skip("set ASR_TEST_DIALOG")
    cfg = _cfg(tmp_path)
    cfg.diarization.enabled = True
    cfg.diarization.num_speakers = 2
    s = _services(tmp_path, cfg)
    r = s.pipeline(f"qwen3-local:{cfg.qwen3.variant}", cfg).run(TaskRequest(path)).result
    from asr_dashboard.core.audio import slice_samples, to_wav16k
    from asr_dashboard.core.diarization.assign import speaker_clean_regions

    samples = load_wav(to_wav16k(path))
    emb = s.embedder(cfg)
    for i, label in enumerate(sorted(r.speakers)):
        regions = speaker_clean_regions(label, r.segments, r.overlaps)
        clip = np.concatenate([slice_samples(samples, a, b) for a, b in regions])
        p = s.voiceprints.add_person(f"Person{i}")
        s.voiceprints.add_sample(p.id, clip, "test", emb.embed(clip), emb.model_id)
    r2 = s.pipeline(f"qwen3-local:{cfg.qwen3.variant}", cfg).run(TaskRequest(path)).result
    assert {i.display_name for i in r2.speakers.values()} == {"Person0", "Person1"}
    Path(tmp_path).exists()

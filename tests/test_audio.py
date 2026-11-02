import shutil

import numpy as np
import pytest
import soundfile as sf

from asr_dashboard.core import audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_to_wav16k_and_probe(tmp_path):
    sr = 44100
    t = np.arange(sr * 2) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 220 * t)], axis=1) * 0.3
    src = tmp_path / "in.flac"
    sf.write(src, stereo, sr)
    info = audio.probe(src)
    assert info.duration == pytest.approx(2.0, abs=0.05)
    out = audio.to_wav16k(src, temp_dir=str(tmp_path))
    data, osr = sf.read(out)
    assert osr == 16000 and data.ndim == 1
    assert len(data) == pytest.approx(32000, abs=200)
    mp3 = audio.encode(out, "mp3", temp_dir=str(tmp_path))
    assert mp3.stat().st_size < out.stat().st_size


def test_corrupted_file_raises(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not really audio" * 100)
    with pytest.raises(audio.AudioError):
        audio.to_wav16k(bad, temp_dir=str(tmp_path))


def test_pcm_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 0.99], dtype=np.float32)
    y = audio.pcm16_to_float(audio.float_to_pcm16(x))
    assert np.allclose(x, y, atol=1e-4)

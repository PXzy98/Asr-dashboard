import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["ASR_DASHBOARD_NO_KEYRING"] = "1"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("ASR_DASHBOARD_HOME", str(home))
    monkeypatch.setenv("ASR_DASHBOARD_CONFIG", str(tmp_path / "config.yaml"))
    yield home


def make_tone(seconds: float, sr: int = 16000, freq: float = 220.0, amp: float = 0.3):
    import numpy as np

    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")

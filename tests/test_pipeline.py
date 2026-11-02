import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from asr_dashboard.config.schema import AppConfig
from asr_dashboard.core.diarization.base import DiarizationOutput, SpeakerTurn
from asr_dashboard.core.pipeline import Pipeline, TaskRequest
from asr_dashboard.core.voiceprints.store import VoiceprintStore
from asr_dashboard.models import CancelledError, CancelToken
from tests.fakes import FakeEngine

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def make_audio(path, seconds=150, sr=16000):
    rng = np.random.default_rng(0)
    x = (0.2 * rng.standard_normal(int(seconds * sr))).astype("float32")
    for a in (55.0, 113.0):  # silences for hybrid cuts
        x[int(a * sr):int((a + 1.0) * sr)] = 0
    sf.write(path, x, sr)
    return path


class FakeDiarizer:
    device = "cpu"
    name = "fake"

    def __init__(self, fail=False):
        self.fail = fail

    def diarize(self, req, progress, cancel):
        if self.fail:
            raise RuntimeError("model missing")
        turns = []
        t, spk = 0.0, 0
        while t < 150:
            turns.append(SpeakerTurn(t, min(150, t + 10), f"spk{spk}"))
            t += 10
            spk ^= 1
        return DiarizationOutput(turns, list(turns), backend="fake")


class FakeEmbedder:
    kind = "wavlm"
    model_id = "fake-emb"

    def embed(self, clip):
        # deterministic: speaker identity encoded by RNG seed of first sample sign (not meaningful) -> constant
        return np.array([1.0, 0.0], dtype=np.float32)


def cfg_for(tmp_path, **kw):
    c = AppConfig()
    c.app.output_dir = str(tmp_path / "out")
    c.app.temp_dir = str(tmp_path / "tmp")
    c.output.formats = ["srt", "txt", "json"]
    for k, v in kw.items():
        sec, _, key = k.partition("__")
        setattr(getattr(c, sec), key, v)
    return c


def test_full_pipeline_fake_engine(tmp_path):
    src = make_audio(tmp_path / "talk.wav")
    eng = FakeEngine(script=lambda c: f"chunk {c.index} says hello.")
    cfg = cfg_for(tmp_path)
    cfg.glossaries = []
    out = Pipeline(cfg, eng, use_silero=False).run(TaskRequest(str(src), task_context="会议：预算"), None, CancelToken())
    r = out.result
    assert len(r.segments) == 3  # hybrid cuts at the two silences
    starts = [s.start for s in r.segments]
    assert starts == sorted(starts) and starts[1] == pytest.approx(55.5, abs=0.2)
    assert all(s.end > s.start for s in r.segments)
    assert r.context_used == "会议：预算" and eng.contexts[0] == "会议：预算"
    assert r.postprocess_mode == "rules"
    assert set(out.outputs) == {"srt", "txt", "json"}
    srt = out.outputs["srt"].read_text(encoding="utf-8")
    assert "00:00:55," in srt
    data = json.loads(out.outputs["json"].read_text(encoding="utf-8"))
    assert data["engine_id"] == "fake" and "识别" in data["timings"]
    assert not list((tmp_path / "tmp").glob("*.wav"))  # temp cleaned


def test_pipeline_diarization_and_voiceprints(tmp_path):
    src = make_audio(tmp_path / "talk.wav")
    eng = FakeEngine(script=lambda c: " ".join(f"w{i}" for i in range(40)), words=True)
    cfg = cfg_for(tmp_path, diarization__enabled=True)
    store = VoiceprintStore(tmp_path / "vp")
    p = store.add_person("王五")
    store.add_sample(p.id, np.zeros(16000, np.float32), embedding=np.array([1.0, 0.0]), model_id="fake-emb")
    pipe = Pipeline(cfg, eng, diarizer_provider=lambda c: FakeDiarizer(), voiceprint_store=store,
                    embedder_provider=lambda c: FakeEmbedder(), use_silero=False)
    r = pipe.run(TaskRequest(str(src)), None, CancelToken()).result
    spk = [s.speaker for s in r.segments]
    assert {"SPEAKER_00", "SPEAKER_01"} <= set(spk)
    # speakers alternate every 10 s -> word-level split produces many alternating segments
    changes = sum(1 for a, b in zip(spk, spk[1:]) if a != b)
    assert changes >= 10
    # both speakers have the same fake embedding, Hungarian gives the person to only one of them
    named = [i for i in r.speakers.values() if i.display_name == "王五"]
    assert len(named) == 1 and named[0].match_score == pytest.approx(1.0)
    assert all(i.embedding for i in r.speakers.values())
    srt = (tmp_path / "out" / "talk.srt").read_text(encoding="utf-8")
    assert "[王五]" in srt and "[SPEAKER_0" in srt


def test_diarization_failure_is_warning(tmp_path):
    src = make_audio(tmp_path / "a.wav", 20)
    infos = []
    pipe = Pipeline(cfg_for(tmp_path, diarization__enabled=True), FakeEngine(),
                    diarizer_provider=lambda c: FakeDiarizer(fail=True), use_silero=False)
    pipe.on_info = infos.append
    out = pipe.run(TaskRequest(str(src)), None, CancelToken())
    assert out.result.segments and any("说话人分离失败" in w for w in out.result.warnings)
    assert infos and "model missing" in infos[0]


def test_streaming_and_cancel(tmp_path):
    src = make_audio(tmp_path / "s.wav")
    cfg = cfg_for(tmp_path, output__streaming=True)
    eng = FakeEngine()
    eng.batch_size = 1
    out = Pipeline(cfg, eng, use_silero=False).run(TaskRequest(str(src)), None, CancelToken())
    assert out.outputs["srt"].read_text(encoding="utf-8").count("-->") == 3
    tok = CancelToken()
    eng2 = FakeEngine(script=lambda c: tok.cancel() or "x")
    eng2.batch_size = 1
    with pytest.raises(CancelledError):
        Pipeline(cfg, eng2, use_silero=False).run(TaskRequest(str(src)), None, tok)


def test_progress_monotonic(tmp_path):
    src = make_audio(tmp_path / "p.wav", 70)
    seen = []
    Pipeline(cfg_for(tmp_path), FakeEngine(), use_silero=False).run(
        TaskRequest(str(src)), lambda s, o, d: seen.append(o), CancelToken())
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)

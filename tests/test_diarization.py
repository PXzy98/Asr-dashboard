import sys
import textwrap

import numpy as np
import pytest

from asr_dashboard.config.schema import DiarizationSection
from asr_dashboard.core.diarization import assign, factory
from asr_dashboard.core.diarization.base import (
    DiarizationOutput,
    DiarizeRequest,
    SpeakerTurn,
    normalize_labels,
)
from asr_dashboard.core.diarization.diarizen_backend import DiarizenDiarizer, make_exclusive
from asr_dashboard.core.diarization.embedding_backend import EmbeddingDiarizer, cluster_embeddings
from asr_dashboard.models import CancelledError, CancelToken, Segment, Word

T = SpeakerTurn


def test_word_level_assignment_splits_inside_sentence():
    words = [Word("hello", 0.0, 0.4), Word("there", 0.5, 0.9), Word("hi", 1.1, 1.3), Word("back", 1.4, 1.8)]
    seg = Segment("hello there hi back", 0.0, 1.8, words=words)
    turns = [T(0.0, 1.0, "A"), T(1.0, 2.0, "B")]
    out, warns = assign.assign_speakers([seg], turns)
    assert [(s.text, s.speaker) for s in out] == [("hello there", "A"), ("hi back", "B")]
    assert out[1].start == 1.1 and out[0].end == 0.9
    assert warns == []


def test_segment_level_assignment_and_warning():
    segs = [Segment("a", 0, 10), Segment("b", 10, 20), Segment("c", 20, 30)]
    turns = [T(0, 9, "A"), T(9, 14, "B"), T(14, 20, "A"), T(20, 30, "B")]
    out, warns = assign.assign_speakers(segs, turns)
    assert [s.speaker for s in out] == ["A", "A", "B"]  # b: A 6 s (60%) vs B 4 s (40%) -> ambiguous
    assert len(warns) == 1 and "强制对齐" in warns[0]


def test_segment_without_overlap_uses_nearest_turn():
    out, _ = assign.assign_speakers([Segment("x", 5.0, 5.5)], [T(0, 4.8, "A"), T(10, 12, "B")])
    assert out[0].speaker == "A"


def test_clean_turns_merge_and_absorb():
    turns = [T(0, 2, "A"), T(2.3, 4, "A"), T(4, 4.1, "B"), T(4.1, 6, "A"), T(6, 9, "B")]
    out = assign.clean_turns(turns, min_turn=0.3, merge_gap=0.5)
    assert [(t.start, t.end, t.speaker) for t in out] == [(0, 6, "A"), (6, 9, "B")]


def test_overlaps():
    turns = [T(0, 5, "A"), T(4, 8, "B"), T(7.5, 9, "C")]
    ov = assign.overlaps_from_turns(turns)
    assert ov[0] == (4, 5, ["A", "B"])
    assert ov[1][2] == ["B", "C"]


def test_make_exclusive_and_normalize():
    turns = [T(0, 5, "x"), T(4, 10, "y")]
    ex = make_exclusive(turns)
    assert [(t.start, t.end, t.speaker) for t in ex] == [(0, 4, "x"), (4, 10, "y")]
    n = normalize_labels(DiarizationOutput(turns, ex, {"y": np.ones(2)}))
    assert {t.speaker for t in n.turns} == {"SPEAKER_00", "SPEAKER_01"}
    assert "SPEAKER_01" in n.embeddings


def test_speaker_clean_regions():
    segs = [Segment("a", 0, 12, speaker="A"), Segment("b", 12, 13, speaker="B"), Segment("a", 13, 30, speaker="A"),
            Segment("a", 30, 31, speaker="A")]
    regions = assign.speaker_clean_regions("A", segs, [(29.5, 30.5, ["A", "B"])], target=20)
    assert regions == [(0, 12), (13, 29.5)]


def _blobs(n_per, dims=16, k=3, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, dims)) * 5
    x = np.concatenate([c + rng.standard_normal((n_per, dims)) * 0.3 for c in centres])
    y = np.repeat(np.arange(k), n_per)
    return x, y


@pytest.mark.parametrize("method", ["agglomerative", "spectral"])
def test_clustering_estimates_count(method):
    x, y = _blobs(20)
    labels = cluster_embeddings(x, 0, 1, 10, method=method, threshold=0.5)
    assert len(set(labels)) == 3
    labels = cluster_embeddings(x, 2, method=method)
    assert len(set(labels)) == 2


def test_embedding_diarizer_with_fake_embedder():
    sr = 16000
    rng = np.random.default_rng(1)
    audio = (0.3 * rng.standard_normal(sr * 12)).astype("float32")

    def embed(clips):
        # first 6 s speaker A, last 6 s speaker B (identified by clip position via mean value trick)
        return np.array([[1.0, 0.0] if c.__array_interface__["data"][0] < audio.__array_interface__["data"][0] + 6 * sr * 4
                          else [0.0, 1.0] for c in clips])

    d = EmbeddingDiarizer("unused", embed_fn=embed, use_silero=False)
    out = d.diarize(DiarizeRequest("x.wav", audio), lambda *a: None, CancelToken())
    assert len({t.speaker for t in out.turns}) == 2
    assert out.turns[0].start == pytest.approx(0, abs=0.1) and out.turns[-1].end == pytest.approx(12, abs=0.1)


FAKE_WORKER = textwrap.dedent('''
    import json, sys, time
    req = json.loads(sys.stdin.readline())
    print(json.dumps({"type": "progress", "value": 0.5, "message": "half"}), flush=True)
    if req["wav"] == "slow.wav":
        time.sleep(30)
    if req["wav"] == "bad.wav":
        print(json.dumps({"type": "error", "message": "boom"}), flush=True); sys.exit(1)
    print("some noise from a library")
    print(json.dumps({"type": "result", "turns": [{"start": 0, "end": 2, "speaker": "spk0"},
        {"start": 1.5, "end": 4, "speaker": "spk1"}], "num": req["num_speakers"]}), flush=True)
''')


@pytest.fixture
def fake_worker(tmp_path):
    p = tmp_path / "worker.py"
    p.write_text(FAKE_WORKER)
    return str(p)


def test_diarizen_subprocess_protocol(fake_worker):
    d = DiarizenDiarizer(None, None, python=sys.executable, script=fake_worker)
    progress = []
    out = d.diarize(DiarizeRequest("ok.wav", None, num_speakers=2), lambda v, m: progress.append(v), CancelToken())
    assert progress == [0.5]
    assert len(out.turns) == 2 and out.exclusive_turns[0].end == pytest.approx(1.5)


def test_diarizen_error_and_cancel(fake_worker):
    d = DiarizenDiarizer(None, None, python=sys.executable, script=fake_worker)
    with pytest.raises(RuntimeError, match="boom"):
        d.diarize(DiarizeRequest("bad.wav", None), lambda *a: None, CancelToken())
    tok = CancelToken()
    import threading
    import time

    threading.Timer(0.5, tok.cancel).start()
    t0 = time.time()
    with pytest.raises(CancelledError):
        d.diarize(DiarizeRequest("slow.wav", None), lambda *a: None, tok)
    assert time.time() - t0 < 5


def test_factory_auto_order(monkeypatch):
    cfg = DiarizationSection()
    monkeypatch.setattr(factory.pyannote_backend, "availability", lambda d: (False, "no model"))
    monkeypatch.setattr(factory.diarizen_backend, "availability", lambda d, a: (True, ""))
    assert factory.choose_backend(cfg).name == "diarizen"
    monkeypatch.setattr(factory.diarizen_backend, "availability", lambda d, a: (False, "no env"))
    monkeypatch.setattr(factory, "_embedding_ok", lambda: (False, "no torch"))
    info = factory.choose_backend(cfg)
    assert not info.available and "no model" in info.reason

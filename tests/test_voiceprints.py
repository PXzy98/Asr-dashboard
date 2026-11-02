import numpy as np
import pytest

from asr_dashboard.core.voiceprints import matcher
from asr_dashboard.core.voiceprints.embedder import check_sample_quality
from asr_dashboard.core.voiceprints.store import VoiceprintStore

M = "test-model"


def unit(*v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def tone(sec=2):
    return (0.1 * np.sin(np.arange(int(16000 * sec)) / 10)).astype("float32")


def test_store_crud(tmp_path):
    s = VoiceprintStore(tmp_path / "vp")
    a = s.add_person("张三", "产品经理")
    smp = s.add_sample(a.id, tone(), "file", unit(1, 0, 0), M)
    s.add_sample(a.id, tone(3), "mic", unit(0.9, 0.1, 0), M)
    p = s.get_person(a.id)
    assert p.name == "张三" and len(p.samples) == 2 and p.total_duration == pytest.approx(5)
    assert smp.file.exists()
    s.update_person(a.id, name="张三丰", note="x")
    assert s.get_person(a.id).name == "张三丰"
    s.delete_sample(smp.id)
    assert not smp.file.exists() and len(s.get_person(a.id).samples) == 1
    with pytest.raises(ValueError):
        s.add_person("  ")
    s.delete_person(a.id)
    assert s.list_people() == []


def test_centroids_and_missing(tmp_path):
    s = VoiceprintStore(tmp_path / "vp")
    a = s.add_person("A")
    s.add_sample(a.id, tone(), embedding=unit(1, 0), model_id=M)
    s.add_sample(a.id, tone())
    cents = s.centroids(M)
    assert len(cents) == 1 and np.allclose(cents[0][2], unit(1, 0))
    assert len(s.missing_embeddings(M)) == 1
    n = s.recompute("other", lambda x: unit(0, 1))
    assert n == 2 and len(s.centroids("other")) == 1


def test_zip_roundtrip(tmp_path):
    s = VoiceprintStore(tmp_path / "vp1")
    a = s.add_person("Alice", "note")
    s.add_sample(a.id, tone(), embedding=unit(1, 2), model_id=M)
    z = s.export_zip(tmp_path / "lib.zip")
    t = VoiceprintStore(tmp_path / "vp2")
    assert t.import_zip(z) == 1
    p = t.list_people()[0]
    assert p.name == "Alice" and p.note == "note" and p.samples[0].file.exists()
    assert np.allclose(t.centroids(M)[0][2], unit(1, 2))
    # importing again merges by name (no duplicate person)
    assert t.import_zip(z) == 0 and len(t.list_people()[0].samples) == 2


def test_hungarian_one_to_one_and_threshold():
    people = [("p1", "张三", unit(1, 0, 0)), ("p2", "李四", unit(0, 1, 0))]
    # both speakers are closest to 张三, but only one may get him
    speakers = {"SPEAKER_00": unit(1, 0.1, 0), "SPEAKER_01": unit(1, 0.6, 0)}
    m = matcher.match_speakers(speakers, people, threshold=0.4)
    assert m["SPEAKER_00"].name == "张三" and m["SPEAKER_01"].name == "李四"
    m = matcher.match_speakers(speakers, people, threshold=0.6)
    assert "SPEAKER_01" not in m and m["SPEAKER_00"].name == "张三"
    # unknown voice stays unmatched
    m = matcher.match_speakers({"SPEAKER_00": unit(0, 0, 1)}, people, threshold=0.6)
    assert m == {}
    assert matcher.match_speakers({}, people, 0.5) == {}


def test_calibration():
    rng = np.random.default_rng(0)
    base = {"a": unit(1, 0, 0, 0), "b": unit(0, 1, 0, 0), "c": unit(0, 0, 1, 0)}
    per = {k: [v + rng.normal(0, 0.05, 4).astype("float32") for _ in range(3)] for k, v in base.items()}
    cal = matcher.calibrate(per)
    assert 0.3 <= cal.threshold <= 0.95 and cal.intra_mean > cal.inter_max
    assert matcher.calibrate({"a": [unit(1, 0)]}).threshold is None


def test_sample_quality_warnings():
    assert any("10 秒" in w for w in check_sample_quality(tone(3)))
    assert any("多个说话人" in w for w in check_sample_quality(tone(12), multi_speaker=True))

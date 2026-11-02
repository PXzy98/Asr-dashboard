import numpy as np
import pytest

from asr_dashboard.core import segmentation as segm
from asr_dashboard.core.segmentation import Bound, SegParams
from asr_dashboard.models import Segment, Word

SR = 16000


def speechy(seconds, sr=SR):
    rng = np.random.default_rng(0)
    return (0.2 * rng.standard_normal(int(seconds * sr))).astype("float32")


def with_silences(total, silences):
    x = speechy(total)
    for a, b in silences:
        x[int(a * SR):int(b * SR)] = 0.0
    return x


def test_duration_strategy_overlap_and_tail():
    p = SegParams(strategy="duration", target_duration=60, max_segment=60, min_segment=5, overlap=0.5)
    b = segm.split_duration(150, p)
    assert b[0].start == 0 and b[0].end == 60 and b[0].overlap_prev == 0
    assert b[1].start == pytest.approx(59.5) and b[1].overlap_prev == pytest.approx(0.5)
    assert b[-1].end == 150
    # short tail is merged into previous chunk
    b2 = segm.split_duration(122, p)
    assert all(x.end - x.start >= 5 for x in b2)
    assert b2[-1].end == 122


def test_hybrid_cuts_in_longest_silence():
    total = 130
    audio = with_silences(total, [(52.0, 52.4), (55.0, 56.5), (110.0, 111.0)])
    p = SegParams(strategy="hybrid", target_duration=60, max_segment=60, min_segment=5)
    b = segm.plan_chunks(audio, p, use_silero=False)
    assert b[0].end == pytest.approx(55.75, abs=0.1)  # middle of the longest silence in [50,60]
    assert b[0].overlap_prev == 0 and b[1].overlap_prev == 0
    assert b[1].start == b[0].end
    assert all(x.end - x.start <= 60 + 1e-6 for x in b)
    assert b[-1].end == pytest.approx(total, abs=0.05)


def test_hybrid_hard_cut_without_silence():
    audio = speechy(150)
    p = SegParams(strategy="hybrid", target_duration=60, max_segment=60, min_segment=5, overlap=0.5)
    b = segm.plan_chunks(audio, p, use_silero=False)
    assert b[0].end == pytest.approx(60)
    assert b[1].start == pytest.approx(59.5) and b[1].overlap_prev == pytest.approx(0.5)


def test_short_audio_single_chunk():
    p = SegParams()
    assert segm.plan_chunks(speechy(10), p, use_silero=False) == [Bound(0.0, 10.0)]


def test_vad_strategy_skips_silence_and_groups():
    regions = [(1, 10), (12, 30), (40, 70), (75, 80), (100, 200)]
    p = SegParams(strategy="vad", target_duration=60, max_segment=60, min_segment=5)
    b = segm.split_vad(220, regions, p, pad=0.0)
    assert b[0].start == 1 and b[0].end == 30  # groups 1-10 and 12-30
    assert b[1].start == 40 and b[1].end == 80
    long = [x for x in b if x.start >= 100]
    assert all(x.end - x.start <= 60 for x in long)
    assert long[1].overlap_prev > 0


def test_make_chunks_offsets():
    audio = speechy(130)
    p = SegParams(strategy="duration")
    chunks = segm.make_chunks(audio, segm.plan_chunks(audio, p, use_silero=False))
    assert chunks[1].offset == pytest.approx(59.5)
    assert len(chunks[0].samples) == 60 * SR


def test_text_dedup_overlap():
    prev = [Segment("今天天气很好，我们去公园", 0, 60)]
    nxt = [Segment("去公园散步吧", 59.5, 120)]
    out = segm.dedup_chunk_segments([prev, nxt], [Bound(0, 60), Bound(59.5, 120, 0.5)])
    assert out[1].text == "散步吧"
    prev = [Segment("we went to the park", 0, 60)]
    nxt = [Segment("the park and played", 59.5, 120)]
    out = segm.dedup_chunk_segments([prev, nxt], [Bound(0, 60), Bound(59.5, 120, 0.5)])
    assert out[1].text == "and played"


def test_no_dedup_without_overlap():
    prev = [Segment("hello world", 0, 60)]
    nxt = [Segment("world peace", 60, 120)]
    out = segm.dedup_chunk_segments([prev, nxt], [Bound(0, 60), Bound(60, 120, 0.0)])
    assert out[1].text == "world peace"


def test_word_level_dedup_by_time():
    w1 = [Word("a", 58.0, 58.5), Word("b", 59.4, 59.6), Word("c", 59.8, 59.95)]
    w2 = [Word("b", 59.45, 59.62), Word("c", 59.8, 59.95), Word("d", 60.2, 60.5)]
    prev = [Segment("a b c", 58.0, 59.95, words=w1)]
    nxt = [Segment("b c d", 59.45, 60.5, words=w2)]
    out = segm.dedup_chunk_segments([prev, nxt], [Bound(0, 60), Bound(59.5, 120, 0.5)])
    allw = [w.text for s in out for w in s.words]
    assert allw == ["a", "b", "c", "d"]

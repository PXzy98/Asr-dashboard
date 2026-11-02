import logging

import numpy as np
import pytest

from asr_dashboard.config.schema import Qwen3Section
from asr_dashboard.engines.qwen3 import common
from asr_dashboard.engines.qwen3 import engine as eng
from asr_dashboard.engines.qwen3.common import RawResult
from asr_dashboard.models import AudioChunk, CancelledError, CancelToken
from tests.fakes import FakeAligner, FakeBackend


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(eng, "backend_status", lambda b: common.BackendStatus(True))
    state = {}

    def make_engine(cfg=None, backend=None, aligner=True):
        cfg = cfg or Qwen3Section()
        e = eng.Qwen3LocalEngine(cfg)
        state["backend"] = backend or FakeBackend()
        monkeypatch.setattr(e, "_make_backend", lambda: state["backend"])
        monkeypatch.setattr(e, "_make_aligner", lambda: FakeAligner())
        return e

    return make_engine


def chunks(n, dur=10.0):
    return [AudioChunk(offset=i * dur, duration=dur, samples=np.zeros(int(dur * 16000), np.float32), index=i)
            for i in range(n)]


def test_settings_schema_covers_all_spec_keys():
    keys = {f.key for f in eng.Qwen3LocalEngine.settings_schema()}
    expected = {"variant", "backend", "source", "local_path", "aligner_local_path", "cache_dir", "device", "dtype",
                "attn_implementation", "torch_compile", "max_inference_batch_size", "max_new_tokens",
                "gpu_memory_utilization", "language", "context", "context_max_chars", "aligner.enabled",
                "aligner.device", "aligner.dtype", "aligner.attn_implementation", "keep_loaded", "oom_fallback_cpu"}
    assert expected <= keys


def test_load_params_logged_and_passed(patched, caplog):
    cfg = Qwen3Section(variant="0.6b", max_new_tokens=2048, max_inference_batch_size=8, device="cpu", dtype="float32")
    e = patched(cfg)
    with caplog.at_level(logging.INFO):
        e.load()
    p = e.backend.params
    assert p.model_path == "Qwen/Qwen3-ASR-0.6B" and p.max_new_tokens == 2048
    assert p.max_inference_batch_size == 8 and p.device == "cpu" and p.dtype == "float32"
    assert p.aligner_path == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert "Qwen3-ASR 加载参数" in caplog.text and "max_new_tokens=2048" in caplog.text


def test_transformers_backend_uses_hf_repos(patched):
    e = patched(Qwen3Section(backend="transformers", device="cpu"))
    e.load()
    assert e.backend.params.model_path == "Qwen/Qwen3-ASR-1.7B-hf"
    assert e.backend.params.aligner_path == "Qwen/Qwen3-ForcedAligner-0.6B-hf"


def test_offsets_and_sentence_split_with_aligner(patched):
    e = patched(Qwen3Section(device="cpu", max_inference_batch_size=2))
    segs = e.transcribe(chunks(3), None, "术语：千问", lambda *a: None, CancelToken())
    assert len(e.backend.calls) == 2 and e.backend.calls[0][1] == ["术语：千问"] * 2
    assert all(s.start > 0 or i == 0 for i, s in enumerate(segs))
    assert segs[1].start >= 10 and segs[2].start >= 20
    assert all(s.end > s.start for s in segs)
    assert segs[0].words and segs[0].text.endswith("。")


def test_no_aligner_uses_chunk_bounds(patched):
    e = patched(Qwen3Section(device="cpu", aligner={"enabled": False}))
    segs = e.transcribe(chunks(2), "zh", None, lambda *a: None, CancelToken())
    assert [(s.start, s.end) for s in segs] == [(0.0, 10.0), (10.0, 20.0)]
    assert e.backend.calls[0][2] == ["Chinese", "Chinese"]


def test_unsupported_language_skips_alignment(patched):
    be = FakeBackend(script=lambda i, w: RawResult("Thai", "สวัสดี"))
    e = patched(Qwen3Section(device="cpu"), backend=be)
    segs = e.transcribe(chunks(1), None, None, lambda *a: None, CancelToken())
    assert (segs[0].start, segs[0].end) == (0.0, 10.0) and not segs[0].words
    assert any("不支持" in w for w in e.warnings)


def test_oom_halves_batch(patched):
    be = FakeBackend(oom_times=1)
    e = patched(Qwen3Section(device="cpu", max_inference_batch_size=4, aligner={"enabled": False}), backend=be)
    segs = e.transcribe(chunks(4), None, None, lambda *a: None, CancelToken())
    assert len(segs) == 4 and e.batch_size == 2


def test_oom_falls_back_to_cpu(patched, monkeypatch):
    monkeypatch.setattr(eng.devices, "resolve_device", lambda d, gpus=None: "cpu" if d == "cpu" else "cuda:0")
    monkeypatch.setattr(eng.devices, "resolve_dtype_name", lambda d, dev: "float32" if dev == "cpu" else "bfloat16")
    be = FakeBackend(oom_until_cpu=True)
    e = patched(Qwen3Section(device="cuda:0", max_inference_batch_size=1, aligner={"enabled": False}), backend=be)
    segs = e.transcribe(chunks(2), None, None, lambda *a: None, CancelToken())
    assert len(segs) == 2 and e.effective_device == "cpu"
    assert e.info_messages and "CPU" in e.info_messages[0]


def test_cancel(patched):
    e = patched(Qwen3Section(device="cpu", max_inference_batch_size=1))
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        e.transcribe(chunks(2), None, None, lambda *a: None, tok)


def test_max_chunk_seconds_capped_for_aligner():
    assert eng.Qwen3LocalEngine(Qwen3Section()).max_chunk_seconds() <= 300
    assert eng.recommended_max_new_tokens(60) >= 480


def test_backend_status_reports_conflict():
    st = common.backend_status("vllm")
    assert isinstance(st.available, bool) and (st.available or st.reason)

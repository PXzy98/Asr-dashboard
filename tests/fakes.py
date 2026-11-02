"""Fake Qwen3 backend / aligner and fake engine for tests."""

import numpy as np

from asr_dashboard.engines.base import BaseASREngine, EngineCapabilities
from asr_dashboard.engines.qwen3.common import LoadParams, Qwen3Backend, RawResult
from asr_dashboard.models import Segment


class OOM(RuntimeError):
    def __init__(self):
        super().__init__("CUDA out of memory. Tried to allocate 2.00 GiB")


class FakeBackend(Qwen3Backend):
    name = "fake"

    def __init__(self, script=None, oom_times=0, oom_until_cpu=False):
        self.script = script or (lambda i, wav: RawResult("Chinese", f"第{i}段。"))
        self.calls = []
        self.oom_times = oom_times
        self.oom_until_cpu = oom_until_cpu
        self.params = None
        self.counter = 0

    def load(self, p: LoadParams):
        self.params = p
        return {"pretrained_model_name_or_path": p.model_path, "device_map": p.device, "dtype": p.dtype,
                "max_inference_batch_size": p.max_inference_batch_size, "max_new_tokens": p.max_new_tokens}

    def set_batch_size(self, n):
        self.bs = n

    def transcribe(self, wavs, contexts, languages):
        self.calls.append((len(wavs), list(contexts), list(languages)))
        if self.oom_times > 0:
            self.oom_times -= 1
            raise OOM()
        if self.oom_until_cpu and self.params.device != "cpu":
            raise OOM()
        out = []
        for w in wavs:
            out.append(self.script(self.counter, w))
            self.counter += 1
        return out

    def unload(self):
        pass


class FakeAligner:
    def __init__(self):
        self.model = object()

    def load(self, *a):
        return {"aligner": True}

    def align(self, wavs, texts, languages):
        out = []
        for w, t in zip(wavs, texts):
            dur = len(w) / 16000
            chars = [c for c in t if c.isalnum()]
            step = dur / max(1, len(chars))
            out.append([(c, i * step, i * step + step * 0.97) for i, c in enumerate(chars)])
        return out

    def unload(self):
        pass


class FakeEngine(BaseASREngine):
    """Returns one segment per chunk; text comes from a script(chunk) callable."""

    id = "fake"
    display_name = "Fake"

    def __init__(self, script=None, words=False):
        super().__init__()
        self.script = script or (lambda c: f"chunk {c.index} text.")
        self.words = words
        self.loaded = False
        self.contexts = []
        self.capabilities = EngineCapabilities(segment_timestamps=True, word_timestamps=words, context_prompt=True,
                                               languages=["zh", "en"], is_local=True)

    @classmethod
    def settings_schema(cls):
        return []

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def transcribe(self, chunks, language, context, on_progress, cancel):
        from asr_dashboard.models import Word

        self.contexts.append(context)
        out = []
        for i, c in enumerate(chunks):
            cancel.raise_if_cancelled()
            text = self.script(c)
            seg = Segment(text, c.offset, c.end, language=language or "en")
            if self.words:
                toks = text.split()
                step = c.duration / max(1, len(toks))
                seg.words = [Word(t, c.offset + k * step, c.offset + (k + 0.9) * step) for k, t in enumerate(toks)]
            out.append(seg)
            on_progress(i + 1, len(chunks), "识别")
        return out


def silence(seconds):
    return np.zeros(int(seconds * 16000), dtype=np.float32)

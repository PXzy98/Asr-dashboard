"""Backend using the official ``qwen-asr`` package (Qwen3ASRModel.from_pretrained, transformers inside).

Verified against qwen-asr 0.0.6 source: ``Qwen3ASRModel.from_pretrained(path, forced_aligner=None,
forced_aligner_kwargs=None, max_inference_batch_size=32, max_new_tokens=512, **kwargs->AutoModel)``
and ``transcribe(audio, context="", language=None, return_time_stamps=False)``; ``context`` becomes the
system message of the chat prompt.
"""

from __future__ import annotations

import logging

import numpy as np

from asr_dashboard.core.devices import torch_dtype
from asr_dashboard.engines.qwen3.common import LoadParams, Qwen3Backend, RawResult, cancel_stopping_criteria

log = logging.getLogger(__name__)


class QwenAsrBackend(Qwen3Backend):
    name = "qwen_asr"

    def __init__(self) -> None:
        self.model = None
        self._cancel = None

    def build_kwargs(self, p: LoadParams) -> dict:
        kw: dict = {
            "dtype": torch_dtype(p.dtype),
            "device_map": p.device,
            "max_inference_batch_size": p.max_inference_batch_size,
            "max_new_tokens": p.max_new_tokens,
        }
        if p.attn_implementation:
            kw["attn_implementation"] = p.attn_implementation
        if p.cache_dir:
            kw["cache_dir"] = p.cache_dir
        return kw

    def load(self, p: LoadParams) -> dict:
        from qwen_asr import Qwen3ASRModel

        kw = self.build_kwargs(p)
        # forced aligner is loaded separately (aligner.py) so that unsupported languages can be skipped
        self.model = Qwen3ASRModel.from_pretrained(p.model_path, **kw)
        self._hook_generate()
        return {"pretrained_model_name_or_path": p.model_path, **kw}

    def _hook_generate(self) -> None:
        """qwen-asr forwards extra generate() kwargs to the HF thinker.generate (verified in 0.0.6), so a
        stopping criterion lets a cancel interrupt a running batch instead of waiting for it to finish."""
        inner = getattr(self.model, "model", None)
        orig = getattr(inner, "generate", None)
        if orig is None:
            return

        def generate(*a, **kw):
            if self._cancel is not None:
                kw.setdefault("stopping_criteria", cancel_stopping_criteria(self._cancel))
            return orig(*a, **kw)

        inner.generate = generate

    def set_batch_size(self, n: int) -> None:
        if self.model is not None:
            self.model.max_inference_batch_size = int(n)

    def transcribe(self, wavs: list[np.ndarray], contexts: list[str], languages: list[str | None]) -> list[RawResult]:
        res = self.model.transcribe(
            audio=[(w, 16000) for w in wavs],
            context=list(contexts),
            language=list(languages) if any(languages) else None,
            return_time_stamps=False,
        )
        return [RawResult(language=r.language or None, text=r.text or "") for r in res]

    def unload(self) -> None:
        self.model = None

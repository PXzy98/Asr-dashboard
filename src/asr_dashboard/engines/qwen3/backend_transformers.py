"""Backend using native HuggingFace Transformers support (>= 5.13) with the ``-hf`` checkpoints.

Verified against transformers 5.22 source: ``Qwen3ASRForConditionalGeneration`` + ``AutoProcessor``;
``processor.apply_transcription_request(audio, language=..., prompt=...)`` puts ``prompt`` (context/hot
words) in the system message and forces the language by prefilling ``language X<asr_text>``.
"""

from __future__ import annotations

import logging

import numpy as np

from asr_dashboard.core.devices import torch_dtype
from asr_dashboard.engines.qwen3.common import LoadParams, Qwen3Backend, RawResult, cancel_stopping_criteria

log = logging.getLogger(__name__)


class TransformersBackend(Qwen3Backend):
    name = "transformers"

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.batch_size = 32
        self.max_new_tokens = 1024
        self._cancel = None

    def load(self, p: LoadParams) -> dict:
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        kw: dict = {"dtype": torch_dtype(p.dtype), "device_map": p.device}
        if p.attn_implementation:
            kw["attn_implementation"] = p.attn_implementation
        if p.cache_dir:
            kw["cache_dir"] = p.cache_dir
        self.processor = AutoProcessor.from_pretrained(p.model_path, cache_dir=p.cache_dir or None)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(p.model_path, **kw)
        self.model.eval()
        if p.torch_compile:
            try:
                self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=False)
            except Exception as e:  # noqa: BLE001
                log.warning("torch.compile 失败，继续使用未编译模型：%s", e)
        self.batch_size = p.max_inference_batch_size
        self.max_new_tokens = p.max_new_tokens
        return {"pretrained_model_name_or_path": p.model_path, **kw, "torch_compile": p.torch_compile,
                "max_new_tokens": p.max_new_tokens, "max_inference_batch_size": p.max_inference_batch_size}

    def set_batch_size(self, n: int) -> None:
        self.batch_size = int(n)

    def transcribe(self, wavs: list[np.ndarray], contexts: list[str], languages: list[str | None]) -> list[RawResult]:
        import torch

        out: list[RawResult] = []
        bs = len(wavs) if self.batch_size in (-1, 0) else max(1, self.batch_size)
        for i in range(0, len(wavs), bs):
            sub_w = wavs[i:i + bs]
            sub_c = [c or None for c in contexts[i:i + bs]]
            sub_l = languages[i:i + bs]
            inputs = self.processor.apply_transcription_request(
                audio=list(sub_w),
                language=sub_l if any(sub_l) else None,
                prompt=sub_c if any(sub_c) else None,
            )
            inputs = inputs.to(self.model.device, self.model.dtype)
            with torch.inference_mode():
                extra = {"stopping_criteria": cancel_stopping_criteria(self._cancel)} if self._cancel else {}
                gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, **extra)
            if self._cancel is not None and self._cancel.cancelled:
                break
            new_tokens = gen[:, inputs["input_ids"].shape[1]:]
            for row, lang in zip(new_tokens, sub_l, strict=True):
                if lang:
                    text = self.processor.decode(row, return_format="transcription_only")
                    out.append(RawResult(lang, text.strip()))
                else:
                    parsed = self.processor.decode(row, return_format="parsed")
                    out.append(RawResult(parsed.get("language"), (parsed.get("transcription") or "").strip()))
        return out

    def unload(self) -> None:
        self.model = None
        self.processor = None

"""Qwen3-ForcedAligner wrappers (qwen-asr package or native transformers)."""

from __future__ import annotations

import logging

import numpy as np

from asr_dashboard.core.devices import torch_dtype

log = logging.getLogger(__name__)

AlignItems = list[tuple[str, float, float]]


class BaseAligner:
    def load(self, path: str, device: str, dtype: str, attn: str | None, cache_dir: str | None) -> dict:
        raise NotImplementedError

    def align(self, wavs: list[np.ndarray], texts: list[str], languages: list[str]) -> list[AlignItems]:
        raise NotImplementedError

    def unload(self) -> None:
        self.model = None


class QwenAsrAligner(BaseAligner):
    """``qwen_asr.Qwen3ForcedAligner`` (verified in qwen-asr 0.0.6: from_pretrained(path, **AutoModel kwargs),
    align(audio, text, language) -> list of results with .items[*].text/.start_time/.end_time in seconds)."""

    def __init__(self) -> None:
        self.model = None

    def load(self, path, device, dtype, attn, cache_dir) -> dict:
        from qwen_asr import Qwen3ForcedAligner

        kw: dict = {"dtype": torch_dtype(dtype), "device_map": device}
        if attn:
            kw["attn_implementation"] = attn
        if cache_dir:
            kw["cache_dir"] = cache_dir
        self.model = Qwen3ForcedAligner.from_pretrained(path, **kw)
        return {"pretrained_model_name_or_path": path, **kw}

    def align(self, wavs, texts, languages):
        res = self.model.align(audio=[(w, 16000) for w in wavs], text=list(texts), language=list(languages))
        out = []
        for r in res:
            items = getattr(r, "items", r)
            out.append([(it.text, float(it.start_time), float(it.end_time)) for it in items])
        return out


class TransformersAligner(BaseAligner):
    """Native transformers (>=5.13): Qwen3ASRForTokenClassification + processor.prepare_forced_aligner_inputs /
    decode_forced_alignment (verified in transformers 5.22 source)."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None

    def load(self, path, device, dtype, attn, cache_dir) -> dict:
        from transformers import AutoProcessor, Qwen3ASRForTokenClassification

        kw: dict = {"dtype": torch_dtype(dtype), "device_map": device}
        if attn:
            kw["attn_implementation"] = attn
        if cache_dir:
            kw["cache_dir"] = cache_dir
        self.processor = AutoProcessor.from_pretrained(path, cache_dir=cache_dir or None)
        self.model = Qwen3ASRForTokenClassification.from_pretrained(path, **kw)
        self.model.eval()
        return {"pretrained_model_name_or_path": path, **kw}

    def align(self, wavs, texts, languages):
        import torch

        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=list(wavs), transcript=list(texts), language=list(languages)
        )
        inputs = inputs.to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        res = self.processor.decode_forced_alignment(
            logits, inputs["input_ids"], word_lists, timestamp_token_id=self.model.config.timestamp_token_id
        )
        return [[(d["text"], float(d["start_time"]), float(d["end_time"])) for d in items] for items in res]

    def unload(self) -> None:
        self.model = None
        self.processor = None

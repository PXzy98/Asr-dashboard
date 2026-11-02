"""vLLM backend through ``Qwen3ASRModel.LLM`` (qwen-asr[vllm]); Linux only.

vLLM starts worker processes with ``spawn``; ``main.py`` keeps the entry point under
``if __name__ == "__main__":`` as required by the qwen-asr README. GPU selection for vLLM is done with
``CUDA_VISIBLE_DEVICES`` (vLLM itself has no device_map).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from asr_dashboard.engines.qwen3.common import LoadParams, Qwen3Backend, RawResult

log = logging.getLogger(__name__)


class VllmBackend(Qwen3Backend):
    name = "vllm"

    def __init__(self) -> None:
        self.model = None

    def load(self, p: LoadParams) -> dict:
        if p.device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = p.device.split(":", 1)[1]
        from qwen_asr import Qwen3ASRModel

        kw: dict = {
            "model": p.model_path,
            "gpu_memory_utilization": p.gpu_memory_utilization,
            "max_inference_batch_size": p.max_inference_batch_size,
            "max_new_tokens": p.max_new_tokens,
            "dtype": p.dtype,
        }
        if p.cache_dir:
            kw["download_dir"] = p.cache_dir
        self.model = Qwen3ASRModel.LLM(**kw)
        return kw

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
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:  # noqa: BLE001
            pass

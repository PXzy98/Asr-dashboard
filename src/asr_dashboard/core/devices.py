"""GPU enumeration, VRAM queries, device/dtype resolution. torch is imported lazily."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int
    is_rocm: bool = False

    @property
    def device(self) -> str:
        return f"cuda:{self.index}"

    @property
    def label(self) -> str:
        kind = "ROCm" if self.is_rocm else "CUDA"
        return f"cuda:{self.index} · {self.name} · {kind} · 可用 {self.free_mb / 1024:.1f} / {self.total_mb / 1024:.1f} GB"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def list_gpus() -> list[GPUInfo]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        rocm = getattr(torch.version, "hip", None) is not None
        out = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:  # noqa: BLE001
                total = torch.cuda.get_device_properties(i).total_memory
                free = total - torch.cuda.memory_reserved(i)
            out.append(GPUInfo(i, name, int(total // 2**20), int(free // 2**20), rocm))
        return out
    except Exception as e:  # noqa: BLE001
        log.debug("GPU enumeration failed: %s", e)
        return []


def resolve_device(device: str, gpus: list[GPUInfo] | None = None) -> str:
    """'auto' -> GPU with the most free VRAM (or cpu)."""
    if device and device != "auto":
        if device.startswith("cuda") and not (gpus if gpus is not None else list_gpus()):
            log.warning("配置的设备 %s 不可用，改用 CPU", device)
            return "cpu"
        return device
    gpus = gpus if gpus is not None else list_gpus()
    if not gpus:
        return "cpu"
    return max(gpus, key=lambda g: g.free_mb).device


def bf16_supported(device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    try:
        import torch

        with torch.cuda.device(int(device.split(":")[1]) if ":" in device else 0):
            return bool(torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        return False


def resolve_dtype_name(dtype: str, device: str) -> str:
    if dtype and dtype != "auto":
        if device == "cpu" and dtype == "float16":
            log.warning("CPU 不支持 float16 推理，改用 float32")
            return "float32"
        return dtype
    if device.startswith("cuda"):
        return "bfloat16" if bf16_supported(device) else "float16"
    return "float32"


def torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def resolve_attn(attn: str, dtype_name: str, device: str) -> str | None:
    """Return the attn_implementation to pass, or None to let transformers decide."""
    if attn == "flash_attention_2":
        if not flash_attn_available():
            log.warning("未安装 flash-attn，注意力实现改用 sdpa")
            return "sdpa"
        if dtype_name not in ("float16", "bfloat16") or not device.startswith("cuda"):
            log.warning("FlashAttention 2 需要 GPU 且精度为 fp16/bf16，改用 sdpa")
            return "sdpa"
        return "flash_attention_2"
    if attn in ("sdpa", "eager"):
        return attn
    return None  # auto


def memory_summary(device: str) -> str:
    if not device.startswith("cuda"):
        return "cpu"
    try:
        import torch

        idx = int(device.split(":")[1]) if ":" in device else 0
        alloc = torch.cuda.memory_allocated(idx) / 2**20
        reserved = torch.cuda.memory_reserved(idx) / 2**20
        return f"{device}: allocated {alloc:.0f} MB, reserved {reserved:.0f} MB"
    except Exception:  # noqa: BLE001
        return device


def empty_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


def is_oom_error(e: BaseException) -> bool:
    name = type(e).__name__
    msg = str(e).lower()
    return "OutOfMemory" in name or "out of memory" in msg or "cuda error: out of memory" in msg or "hip out of memory" in msg

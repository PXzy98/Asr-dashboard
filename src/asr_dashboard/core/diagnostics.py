"""Start-up self check shown in Settings -> 环境诊断 (copyable)."""

from __future__ import annotations

import importlib.metadata as md
import platform
import sys


def _v(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "未安装"


def run(cfg=None, secrets=None) -> str:
    from asr_dashboard.core import audio, devices
    from asr_dashboard.engines.qwen3.common import backend_status

    lines = [f"Python {sys.version.split()[0]} · {platform.platform()}"]
    ff = audio.ffmpeg_version()
    lines.append(f"FFmpeg：{ff or '未找到（请安装，见 README）'}")
    try:
        import torch

        hip = getattr(torch.version, "hip", None)
        lines.append(f"PyTorch {torch.__version__} · CUDA {torch.version.cuda or '-'} · ROCm {hip or '-'} · "
                     f"GPU 可用：{torch.cuda.is_available()}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"PyTorch：不可用（{e}）")
    gpus = devices.list_gpus()
    for g in gpus:
        lines.append(f"  {g.label}")
    if not gpus:
        lines.append("  未检测到 GPU，将使用 CPU")
    for b in ("qwen_asr", "transformers", "vllm"):
        st = backend_status(b)
        lines.append(f"后端 {b}：{'可用 ' + st.version if st.available else '不可用 - ' + st.reason}"
                     + ("" if st.available or not st.install else f"（安装：{st.install}）"))
    lines.append(f"flash-attn：{_v('flash-attn')} · transformers：{_v('transformers')} · qwen-asr：{_v('qwen-asr')}")
    lines.append(f"pyannote.audio：{_v('pyannote.audio')} · silero-vad：{_v('silero-vad')}")
    if cfg is not None:
        from asr_dashboard.core.diarization.factory import backend_infos

        for i in backend_infos(cfg.diarization):
            lines.append(f"说话人分离 {i.name}：{'可用' if i.available else '不可用 - ' + i.reason}")
    from asr_dashboard.realtime.capture import loopback_available

    lines.append(f"系统声音采集（WASAPI 回环）：{'可用' if loopback_available() else '不可用（仅 Windows + PyAudioWPatch）'}")
    if secrets is not None:
        lines.append(f"密钥存储：{secrets.backend_name}" + ("（明文）" if secrets.is_plaintext else ""))
    return "\n".join(lines)

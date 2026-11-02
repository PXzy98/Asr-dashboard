"""Local Qwen3-ASR engine (1.7B / 0.6B) with optional Qwen3-ForcedAligner word timestamps."""

from __future__ import annotations

import logging
import os
import threading

from asr_dashboard.config.schema import QWEN_LANGUAGES, Qwen3Section
from asr_dashboard.core import devices
from asr_dashboard.core.sentences import attach_punctuation, split_sentences
from asr_dashboard.engines.base import BaseASREngine, EngineCapabilities, ProgressFn
from asr_dashboard.engines.qwen3 import common
from asr_dashboard.engines.qwen3.common import (
    ALIGNER_MAX_SECONDS,
    ALIGNER_REPO,
    DIALECTS,
    REPO_IDS,
    LoadParams,
    aligner_supports,
    backend_status,
    lang_code,
    lang_name,
)
from asr_dashboard.models import AudioChunk, CancelToken, Segment
from asr_dashboard.settings_fields import SettingField

log = logging.getLogger(__name__)

_LANG_CHOICES = [("auto", "自动识别")] + [(c, f"{n} ({c})") for c, n in QWEN_LANGUAGES.items()]
_DEVICE_CHOICES_STATIC = [("auto", "自动（可用显存最大的显卡）"), ("cpu", "CPU")]


def device_choices() -> list[tuple[str, str]]:
    return [_DEVICE_CHOICES_STATIC[0]] + [(g.device, g.label) for g in devices.list_gpus()] + [_DEVICE_CHOICES_STATIC[1]]


def recommended_max_new_tokens(chunk_seconds: float) -> int:
    """About 8 tokens per second of audio, with head-room, clamped to the allowed range."""
    return int(min(8192, max(64, round(chunk_seconds * 8 * 1.25 / 64) * 64)))


class Qwen3LocalEngine(BaseASREngine):
    id = "qwen3-local"
    display_name = "Qwen3-ASR（本地）"

    def __init__(self, cfg: Qwen3Section, sentence_max_duration: float = 15.0) -> None:
        super().__init__()
        self.cfg = cfg.model_copy(deep=True)
        self.sentence_max_duration = sentence_max_duration
        self.backend: common.Qwen3Backend | None = None
        self.aligner = None
        self._loaded_signature: tuple | None = None
        self._lock = threading.RLock()
        self.load_kwargs: dict = {}
        self.aligner_kwargs: dict = {}
        self.effective_device = "cpu"
        self.batch_size = cfg.max_inference_batch_size
        self.display_name = f"Qwen3-ASR {cfg.variant.upper()}（本地）"

    # ------------------------------------------------------------------ capabilities / schema
    @property
    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            word_timestamps=self.cfg.aligner.enabled, segment_timestamps=True, language_detect=True,
            languages=list(QWEN_LANGUAGES), context_prompt=True, max_audio_seconds=0, max_file_mb=0,
            supports_batch=True, is_local=True, forced_alignment=True,
        )

    @classmethod
    def settings_schema(cls) -> list[SettingField]:
        st = {b: backend_status(b) for b in ("qwen_asr", "transformers", "vllm")}

        def blabel(b: str, text: str) -> str:
            return text if st[b].available else f"{text}（不可用：{st[b].reason}）"

        fa = devices.flash_attn_available()
        return [
            SettingField("variant", "模型规格", "enum", "1.7b",
                         "1.7B 精度最高；0.6B 速度快、占用资源少", choices=[("1.7b", "Qwen3-ASR-1.7B"), ("0.6b", "Qwen3-ASR-0.6B")]),
            SettingField("backend", "推理后端", "enum", "qwen_asr",
                         "qwen_asr：官方 qwen-asr 包；transformers：Transformers 原生（-hf 模型，需 transformers>=5.13）；"
                         "vllm：仅 Linux。\n" + "\n".join(f"{k}: 安装命令 {v.install}" for k, v in st.items() if not v.available and v.install),
                         choices=[("qwen_asr", blabel("qwen_asr", "qwen-asr（官方包）")),
                                  ("transformers", blabel("transformers", "Transformers 原生")),
                                  ("vllm", blabel("vllm", "vLLM（Linux）"))],
                         disabled_choices={k: f"{v.reason}；安装：{v.install}" for k, v in st.items() if not v.available}),
            SettingField("source", "模型来源", "enum", "huggingface", "从哪里下载模型权重",
                         choices=[("huggingface", "HuggingFace"), ("modelscope", "ModelScope（魔搭）"), ("local", "本地目录")]),
            SettingField("local_path", "本地模型路径", "path", "", "Qwen3-ASR 模型目录", visible_if={"source": "local"}),
            SettingField("aligner_local_path", "本地对齐器路径", "path", "", "Qwen3-ForcedAligner 模型目录",
                         visible_if={"source": "local"}),
            SettingField("cache_dir", "模型缓存目录", "path", "", "留空使用默认位置；会设置 HF_HOME / MODELSCOPE_CACHE"),
            SettingField("device", "设备", "enum", "auto", "auto 选择可用显存最大的显卡", choices=device_choices()),
            SettingField("dtype", "精度", "enum", "auto", "auto：GPU 用 bf16（不支持则 fp16），CPU 用 fp32",
                         choices=[("auto", "自动"), ("bfloat16", "bfloat16"), ("float16", "float16"), ("float32", "float32")]),
            SettingField("attn_implementation", "注意力实现", "enum", "auto",
                         "FlashAttention 2 需要已安装 flash-attn，且精度为 fp16/bf16" + ("" if fa else "（当前未安装 flash-attn）"),
                         choices=[("auto", "自动"), ("sdpa", "sdpa"), ("eager", "eager"),
                                  ("flash_attention_2", "flash_attention_2" + ("" if fa else "（未安装）"))],
                         disabled_choices={} if fa else {"flash_attention_2": "未安装 flash-attn：uv sync --extra flash-attn"}),
            SettingField("torch_compile", "torch.compile 加速（实验性）", "bool", False, "仅 transformers 后端",
                         visible_if={"backend": "transformers"}),
            SettingField("max_inference_batch_size", "最大推理批大小", "int", 32, "-1 表示不限；调小可避免显存不足",
                         minimum=1, maximum=256, special_values={-1: "不限"}),
            SettingField("max_new_tokens", "最大生成 token 数", "int", 1024,
                         "长音频需要调大；约每秒音频 8 个 token（60 秒分段推荐 "
                         f"{recommended_max_new_tokens(60)}）", minimum=64, maximum=8192),
            SettingField("gpu_memory_utilization", "vLLM 显存占比", "float", 0.7, "", minimum=0.1, maximum=0.95,
                         step=0.05, visible_if={"backend": "vllm"}),
            SettingField("language", "识别语言", "enum", "auto",
                         f"另支持 22 种中文方言（{DIALECTS}），由模型自动识别，无需选择", choices=_LANG_CHOICES),
            SettingField("context", "默认上下文", "text", "",
                         "专有名词、人名、术语、背景说明。属于“软引导”，不能保证每个词都被识别正确"),
            SettingField("context_max_chars", "上下文最大长度", "int", 2000, "超过时截断并提示", minimum=100, maximum=20000),
            SettingField("aligner.enabled", "启用强制对齐（词级时间戳）", "bool", True,
                         "Qwen3-ForcedAligner-0.6B，支持 11 种语言，单段最长 5 分钟（本程序按 180 秒切块）"),
            SettingField("aligner.device", "对齐器设备", "enum", "follow", "",
                         choices=[("follow", "跟随主模型")] + device_choices(), visible_if={"aligner.enabled": True}),
            SettingField("aligner.dtype", "对齐器精度", "enum", "follow", "",
                         choices=[("follow", "跟随主模型"), ("auto", "自动"), ("bfloat16", "bfloat16"),
                                  ("float16", "float16"), ("float32", "float32")], visible_if={"aligner.enabled": True}),
            SettingField("aligner.attn_implementation", "对齐器注意力实现", "enum", "follow", "",
                         choices=[("follow", "跟随主模型"), ("auto", "自动"), ("sdpa", "sdpa"), ("eager", "eager"),
                                  ("flash_attention_2", "flash_attention_2")], visible_if={"aligner.enabled": True}),
            SettingField("keep_loaded", "模型常驻内存", "bool", True, "关闭时每个任务结束后卸载模型并清理显存；批量处理期间始终常驻"),
            SettingField("oom_fallback_cpu", "显存不足时回退 CPU", "bool", True, "回退时会在界面上提示"),
        ]

    # ------------------------------------------------------------------ model paths
    def _model_paths(self) -> tuple[str, str]:
        c = self.cfg
        native = c.backend == "transformers"
        if c.source == "local":
            if not c.local_path:
                raise RuntimeError("模型来源为“本地”，但没有设置本地模型路径")
            return c.local_path, c.aligner_local_path or ALIGNER_REPO[native]
        repo = REPO_IDS[(c.variant, native)]
        arepo = ALIGNER_REPO[native]
        if c.source == "modelscope":
            from asr_dashboard.engines.qwen3.model_manager import modelscope_download

            return modelscope_download(repo, c.cache_dir or None), (
                modelscope_download(arepo, c.cache_dir or None) if c.aligner.enabled else arepo)
        return repo, arepo

    def _apply_cache_env(self) -> None:
        if self.cfg.cache_dir:
            os.environ["HF_HOME"] = self.cfg.cache_dir
            os.environ["MODELSCOPE_CACHE"] = self.cfg.cache_dir

    def build_load_params(self, force_cpu: bool = False) -> LoadParams:
        c = self.cfg
        device = "cpu" if force_cpu else devices.resolve_device(c.device)
        dtype = devices.resolve_dtype_name("float32" if force_cpu else c.dtype, device)
        attn = devices.resolve_attn(c.attn_implementation, dtype, device)
        a = c.aligner
        a_dev = device if (force_cpu or a.device == "follow") else devices.resolve_device(a.device)
        a_dtype = dtype if a.dtype == "follow" else devices.resolve_dtype_name(a.dtype, a_dev)
        if force_cpu:
            a_dtype = "float32"
        a_attn = attn if a.attn_implementation == "follow" else devices.resolve_attn(a.attn_implementation, a_dtype, a_dev)
        model_path, aligner_path = self._model_paths()
        return LoadParams(
            model_path=model_path, aligner_path=aligner_path if a.enabled else None, device=device, dtype=dtype,
            attn_implementation=attn, aligner_device=a_dev, aligner_dtype=a_dtype, aligner_attn=a_attn,
            max_inference_batch_size=self.batch_size, max_new_tokens=c.max_new_tokens,
            gpu_memory_utilization=c.gpu_memory_utilization, torch_compile=c.torch_compile and c.backend == "transformers",
            cache_dir=c.cache_dir or None,
        )

    def _make_backend(self) -> common.Qwen3Backend:
        b = self.cfg.backend
        st = backend_status(b)
        if not st.available:
            raise RuntimeError(f"推理后端 {b} 不可用：{st.reason}。安装：{st.install}")
        if b == "qwen_asr":
            from asr_dashboard.engines.qwen3.backend_qwen_asr import QwenAsrBackend

            return QwenAsrBackend()
        if b == "transformers":
            from asr_dashboard.engines.qwen3.backend_transformers import TransformersBackend

            return TransformersBackend()
        from asr_dashboard.engines.qwen3.backend_vllm import VllmBackend

        return VllmBackend()

    def _make_aligner(self):
        if self.cfg.backend == "transformers":
            from asr_dashboard.engines.qwen3.aligner import TransformersAligner

            return TransformersAligner()
        from asr_dashboard.engines.qwen3.aligner import QwenAsrAligner

        return QwenAsrAligner()

    # ------------------------------------------------------------------ lifecycle
    def _signature(self) -> tuple:
        c = self.cfg
        return (c.variant, c.backend, c.source, c.local_path, c.aligner_local_path, c.cache_dir, c.device, c.dtype,
                c.attn_implementation, c.torch_compile, c.max_new_tokens, c.gpu_memory_utilization,
                c.aligner.model_dump_json())

    @property
    def is_loaded(self) -> bool:
        return self.backend is not None and self._loaded_signature == self._signature()

    def update_config(self, cfg: Qwen3Section) -> None:
        """Apply new settings; the model is reloaded lazily if a load-relevant setting changed."""
        self.cfg = cfg.model_copy(deep=True)
        self.batch_size = cfg.max_inference_batch_size
        if self.backend is not None:
            self.backend.set_batch_size(self.batch_size)
        self.display_name = f"Qwen3-ASR {cfg.variant.upper()}（本地）"

    def load(self, force_cpu: bool = False) -> None:
        with self._lock:
            if self.is_loaded and not force_cpu:
                return
            self.unload()
            self._apply_cache_env()
            p = self.build_load_params(force_cpu=force_cpu)
            backend = self._make_backend()
            try:
                self.load_kwargs = backend.load(p)
                log.info("Qwen3-ASR 加载参数（%s 后端）：%s", backend.name, _fmt_kwargs(self.load_kwargs))
                if p.aligner_path:
                    aligner = self._make_aligner()
                    self.aligner_kwargs = aligner.load(p.aligner_path, p.aligner_device, p.aligner_dtype,
                                                       p.aligner_attn, p.cache_dir)
                    log.info("Qwen3-ForcedAligner 加载参数：%s", _fmt_kwargs(self.aligner_kwargs))
                    self.aligner = aligner
            except Exception as e:
                backend.unload()
                self.aligner = None
                devices.empty_cache()
                if devices.is_oom_error(e) and not force_cpu and self.cfg.oom_fallback_cpu and p.device != "cpu":
                    msg = f"加载模型时显存不足（{p.device}），已自动回退到 CPU 运行，速度会明显变慢"
                    log.warning(msg)
                    self.info_messages.append(msg)
                    self.warnings.append(msg)
                    self.load(force_cpu=True)
                    return
                raise
            self.backend = backend
            self.effective_device = p.device
            self._loaded_signature = self._signature()
            log.info("显存占用：%s", devices.memory_summary(p.device))

    def unload(self) -> None:
        with self._lock:
            if self.backend is not None:
                self.backend.unload()
            if self.aligner is not None:
                self.aligner.unload()
            self.backend = None
            self.aligner = None
            self._loaded_signature = None
            devices.empty_cache()

    def describe_settings(self) -> dict:
        d = self.cfg.model_dump(mode="json")
        d.pop("context", None)
        d["effective_device"] = self.effective_device
        d["load_kwargs"] = {k: str(v) for k, v in self.load_kwargs.items()}
        return d

    def max_chunk_seconds(self) -> float | None:
        return ALIGNER_MAX_SECONDS if self.cfg.aligner.enabled else common.ASR_MAX_SECONDS

    def reduce_batch_size(self) -> bool:
        cur = self.batch_size if self.batch_size > 0 else 64
        if cur <= 1:
            return False
        self.batch_size = max(1, cur // 2)
        if self.backend is not None:
            self.backend.set_batch_size(self.batch_size)
        log.warning("显存不足，推理批大小减半为 %d", self.batch_size)
        return True

    # ------------------------------------------------------------------ inference
    def transcribe(self, chunks: list[AudioChunk], language: str | None, context: str | None,
                   on_progress: ProgressFn, cancel: CancelToken) -> list[Segment]:
        if not self.is_loaded:
            self.load()
        forced = lang_name(language or self.cfg.language)
        n = len(chunks)
        raws: list[common.RawResult | None] = [None] * n
        bs = n if self.batch_size in (-1, 0) else max(1, self.batch_size)
        i = 0
        while i < n:
            cancel.raise_if_cancelled()
            batch = chunks[i:i + bs]
            self.backend.set_cancel(cancel)
            try:
                res = self.backend.transcribe([c.waveform() for c in batch], [context or ""] * len(batch),
                                              [forced] * len(batch))
            except Exception as e:
                if devices.is_oom_error(e):
                    devices.empty_cache()
                    if bs > 1:
                        bs = max(1, bs // 2)
                        self.batch_size = bs
                        self.backend.set_batch_size(bs)
                        log.warning("识别时显存不足，批大小减为 %d 后重试", bs)
                        continue
                    if self.cfg.oom_fallback_cpu and self.effective_device != "cpu":
                        msg = "识别时显存不足，已自动回退到 CPU 继续运行，速度会明显变慢"
                        self.info_messages.append(msg)
                        self.warnings.append(msg)
                        self.load(force_cpu=True)
                        continue
                raise
            cancel.raise_if_cancelled()  # generation may have been stopped early by the cancel token
            for k, r in enumerate(res):
                raws[i + k] = r
            i += len(batch)
            on_progress(i, n, f"识别 {i}/{n}")
        segments: list[Segment] = []
        align_todo = []
        for c, r in zip(chunks, raws, strict=True):
            r = r or common.RawResult(None, "")
            lang = r.language or forced
            if self.aligner is not None and self.cfg.aligner.enabled and r.text.strip():
                if aligner_supports(lang):
                    align_todo.append((c, r, lang))
                    continue
                self.warnings.append(
                    f"{_fmt_t(c.offset)}–{_fmt_t(c.end)} 识别语言为 {lang or '未知'}，强制对齐器不支持，"
                    "该段改用分段边界作为时间戳")
            if r.text.strip():
                segments.append(Segment(r.text.strip(), c.offset, c.end, language=lang_code(lang)))
        if align_todo:
            on_progress(n, n, "对齐")
            segments.extend(self._align(align_todo, on_progress, cancel))
        segments.sort(key=lambda s: s.start)
        return segments

    def _align(self, todo, on_progress: ProgressFn, cancel: CancelToken) -> list[Segment]:
        out: list[Segment] = []
        bs = len(todo) if self.batch_size in (-1, 0) else max(1, self.batch_size)
        done = 0
        for i in range(0, len(todo), bs):
            cancel.raise_if_cancelled()
            part = todo[i:i + bs]
            try:
                results = self.aligner.align([c.waveform() for c, _, _ in part], [r.text for _, r, _ in part],
                                             [lang for _, _, lang in part])
            except Exception as e:  # alignment failure must not lose the transcript
                log.exception("强制对齐失败")
                self.warnings.append(f"强制对齐失败（{e}），这些段落改用分段边界作为时间戳")
                results = [None] * len(part)
            for (c, r, lang), items in zip(part, results, strict=True):
                if not items:
                    out.append(Segment(r.text.strip(), c.offset, c.end, language=lang_code(lang)))
                    continue
                shifted = [(t, s + c.offset, e + c.offset) for t, s, e in items]
                words = attach_punctuation(r.text, shifted)
                for w in words:  # clamp into the chunk
                    w.start = min(max(w.start, c.offset), c.end)
                    w.end = min(max(w.end, w.start), c.end)
                sents = split_sentences(words, self.sentence_max_duration, language=lang_code(lang))
                out.extend(sents or [Segment(r.text.strip(), c.offset, c.end, language=lang_code(lang))])
            done += len(part)
            on_progress(done, len(todo), f"对齐 {done}/{len(todo)}")
        return out


def _fmt_t(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_kwargs(kw: dict) -> str:
    return ", ".join(f"{k}={v!s}" for k, v in kw.items())

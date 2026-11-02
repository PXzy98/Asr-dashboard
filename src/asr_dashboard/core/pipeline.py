"""Engine-agnostic pipeline:
preprocess -> segment -> recognise (-> align) -> diarize -> voiceprint match -> post-process -> export.

The pipeline only talks to ``BaseASREngine``; there is no engine-specific branching here.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asr_dashboard.config.schema import AppConfig
from asr_dashboard.core import audio as audiomod
from asr_dashboard.core import segmentation as segm
from asr_dashboard.core.context import build_context
from asr_dashboard.core.diarization import assign as dassign
from asr_dashboard.core.diarization.base import DiarizeRequest, normalize_labels
from asr_dashboard.engines.base import BaseASREngine
from asr_dashboard.export.streaming import StreamingWriter
from asr_dashboard.export.writers import ExportOptions, save_outputs
from asr_dashboard.models import (
    AudioChunk,
    CancelledError,
    CancelToken,
    Segment,
    SpeakerInfo,
    TranscriptionResult,
)

log = logging.getLogger(__name__)

# overall progress weights per stage
STAGES = [("preprocess", "预处理", 0.04), ("load", "加载模型", 0.04), ("segment", "分段", 0.02),
          ("recognize", "识别", 0.60), ("diarize", "说话人分离", 0.15), ("voiceprint", "声纹匹配", 0.05),
          ("postprocess", "文本后处理", 0.08), ("export", "导出", 0.02)]
STAGE_LABEL = {k: v for k, v, _ in STAGES}

ProgressCb = Callable[[str, float, str], None]  # (stage, overall 0..1, detail)


@dataclass
class TaskRequest:
    source: str
    task_context: str = ""
    glossary_ids: list[str] = field(default_factory=list)
    output_dir: str | None = None
    output_stem: str | None = None
    on_existing: str | None = None  # override cfg.output.on_existing (batch)
    batch_id: str | None = None
    language: str | None = None  # "auto" / ISO code; None = engine default


@dataclass
class PipelineOutput:
    result: TranscriptionResult
    outputs: dict[str, Path]
    info_messages: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(self, cfg: AppConfig, engine: BaseASREngine, *, secrets=None, diarizer_provider=None,
                 voiceprint_store=None, embedder_provider=None, llm_client=None, use_silero: bool = True,
                 keep_engine_loaded: bool | None = None) -> None:
        self.cfg = cfg
        self.engine = engine
        self.secrets = secrets
        self.diarizer_provider = diarizer_provider  # callable(cfg) -> BaseDiarizer
        self.voiceprint_store = voiceprint_store
        self.embedder_provider = embedder_provider  # callable(cfg) -> VoiceprintEmbedder
        self.llm_client = llm_client
        self.use_silero = use_silero
        self.keep_engine_loaded = keep_engine_loaded
        self.info_messages: list[str] = []
        self.on_info: Callable[[str], None] | None = None
        self.language_override: str | None = None

    # ------------------------------------------------------------------ helpers
    def _info(self, msg: str) -> None:
        self.info_messages.append(msg)
        if self.on_info:
            try:
                self.on_info(msg)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _overall(stage: str, frac: float) -> float:
        done = 0.0
        for key, _label, w in STAGES:
            if key == stage:
                return min(1.0, done + w * max(0.0, min(1.0, frac)))
            done += w
        return done

    # ------------------------------------------------------------------ main entry
    def run(self, req: TaskRequest, progress: ProgressCb | None = None,
            cancel: CancelToken | None = None) -> PipelineOutput:
        cfg = self.cfg
        cancel = cancel or CancelToken()
        prog = progress or (lambda *_: None)
        self.language_override = req.language
        self.info_messages = []

        def step(stage: str, frac: float, detail: str = "") -> None:
            prog(stage, self._overall(stage, frac), detail or STAGE_LABEL.get(stage, stage))

        timings: dict[str, float] = {}
        warnings: list[str] = []
        t_all = time.perf_counter()
        src = Path(req.source)
        out_dir = Path(req.output_dir or cfg.app.output_dir)
        stem = req.output_stem or src.stem
        wav_path: Path | None = None

        try:
            # 1. preprocess ------------------------------------------------------------
            step("preprocess", 0.0, "转换音频（FFmpeg）")
            t = time.perf_counter()
            wav_path = audiomod.to_wav16k(src, cfg.app.temp_dir or None, cancel)
            samples = audiomod.load_wav(wav_path)
            duration = len(samples) / audiomod.SAMPLE_RATE
            timings["预处理"] = time.perf_counter() - t
            if duration < 0.1:
                raise audiomod.AudioError(f"{src.name} 没有可识别的音频")
            cancel.raise_if_cancelled()

            # 2. context ---------------------------------------------------------------
            gls = [g for g in (cfg.glossary(i) for i in req.glossary_ids) if g]
            ctx = build_context(cfg.qwen3.context, gls, req.task_context, cfg.qwen3.context_max_chars)
            warnings += ctx.warnings
            context = ctx.text or None
            caps = self.engine.capabilities
            if context and not caps.context_prompt:
                warnings.append("当前引擎不支持上下文提示，本次未使用上下文")
                context = None

            # 3. load ------------------------------------------------------------------
            step("load", 0.0, "加载模型")
            t = time.perf_counter()
            self.engine.load()
            timings["模型加载"] = time.perf_counter() - t
            cancel.raise_if_cancelled()

            # 4. segmentation -------------------------------------------------------------
            step("segment", 0.0, "分段")
            t = time.perf_counter()
            chunks, bounds = self._plan(samples, wav_path)
            timings["分段"] = time.perf_counter() - t

            # 5. recognition (+ streaming) -----------------------------------------------
            t = time.perf_counter()
            stream = None
            if cfg.output.streaming:
                stream = StreamingWriter(out_dir, stem, cfg.output.formats, ExportOptions.from_config(cfg.output),
                                         overwrite=(req.on_existing or cfg.output.on_existing) == "overwrite")
            segments = self._recognize(chunks, bounds, context, step, cancel, stream)
            timings["识别"] = time.perf_counter() - t
            for w in self.engine.warnings:
                if w not in warnings:
                    warnings.append(w)
            for m in self.engine.info_messages:
                self._info(m)
            self.engine.warnings.clear()
            self.engine.info_messages.clear()
            if any(s.end <= 0 and s.start <= 0 for s in segments) and duration > 0:
                warnings.append("部分分段没有有效时间戳")

            result = TranscriptionResult(
                source_file=str(src), segments=segments, text="", language=_majority_lang(segments),
                duration=duration, engine_id=self.engine.id, engine_settings=self.engine.describe_settings(),
                context_used=context, timings=timings, warnings=warnings,
            )

            # 6. diarization ---------------------------------------------------------------
            diar = None
            if cfg.diarization.enabled:
                t = time.perf_counter()
                diar = self._diarize(result, wav_path, samples, step, cancel)
                timings["说话人分离"] = time.perf_counter() - t

            # 7. voiceprints ---------------------------------------------------------------
            if diar is not None and cfg.voiceprint.enabled and result.speakers:
                t = time.perf_counter()
                self._voiceprints(result, diar, samples, step, cancel)
                timings["声纹匹配"] = time.perf_counter() - t

            # 8. post-processing ----------------------------------------------------------
            t = time.perf_counter()
            step("postprocess", 0.0, "文本后处理")
            from asr_dashboard.config import secrets as sec
            from asr_dashboard.text.postprocess import run_postprocess

            key = self.secrets.get(sec.LLM_KEY) if self.secrets else ""
            run_postprocess(result, cfg, key, context or "", lambda f, m: step("postprocess", f, m), cancel,
                            client=self.llm_client)
            timings["文本后处理"] = time.perf_counter() - t

            # 9. export -------------------------------------------------------------------
            outputs: dict[str, Path] = {}
            step("export", 0.0, "导出")
            t = time.perf_counter()
            if stream is not None:
                result.timings = timings
                stream.rewrite_final(result)
                outputs.update(stream.finalize(result))
            if cfg.output.auto_save:
                remaining = [f for f in cfg.output.formats if f not in outputs] + (["md"] if result.summary else [])
                outputs.update(save_outputs(result, out_dir, stem, remaining, ExportOptions.from_config(cfg.output),
                                            req.on_existing or "rename"))
            timings["导出"] = time.perf_counter() - t
            timings["总计"] = time.perf_counter() - t_all
            step("export", 1.0, "完成")
            return PipelineOutput(result, outputs, list(self.info_messages))
        finally:
            keep = self.keep_engine_loaded
            if keep is None:
                keep = getattr(getattr(self.engine, "cfg", None), "keep_loaded", True)
            if not keep:
                try:
                    self.engine.unload()
                except Exception:  # noqa: BLE001
                    log.exception("卸载模型失败")
            if wav_path is not None and cfg.app.cleanup_temp:
                audiomod.safe_unlink(wav_path)

    # ------------------------------------------------------------------ stages
    def _plan(self, samples: np.ndarray, wav_path: Path) -> tuple[list[AudioChunk], list[segm.Bound]]:
        total = len(samples) / audiomod.SAMPLE_RATE
        if not self.engine.wants_chunked_input():
            return [AudioChunk(offset=0.0, duration=total, path=wav_path, samples=samples)], [segm.Bound(0.0, total)]
        caps = self.engine.capabilities
        cap = self.engine.max_chunk_seconds()
        if caps.max_audio_seconds:
            cap = min(cap or caps.max_audio_seconds, caps.max_audio_seconds)
        params = segm.SegParams.from_config(self.cfg.segmentation, cap)
        bounds = segm.plan_chunks(samples, params, use_silero=self.use_silero)
        return segm.make_chunks(samples, bounds), bounds

    def _recognize(self, chunks, bounds, context, step, cancel, stream) -> list[Segment]:
        n = len(chunks)
        group = max(1, getattr(self.engine, "batch_size", 8) if getattr(self.engine, "batch_size", 8) > 0 else n)
        per_chunk: list[list[Segment]] = [[] for _ in range(n)]
        written = 0
        done = 0
        for g0 in range(0, n, group):
            cancel.raise_if_cancelled()
            part = chunks[g0:g0 + group]

            def on_prog(i: int, tot: int, msg: str, base=g0) -> None:
                if msg.startswith("识别"):
                    step("recognize", (base + i) / n, f"识别 {base + i}/{n}")
                else:
                    step("recognize", (base + min(i, tot)) / n, msg)

            segs = self.engine.transcribe(part, self._language(), context, on_prog, cancel)
            for s in segs:
                idx = _chunk_index(s, part, g0)
                per_chunk[idx].append(s)
            done += len(part)
            step("recognize", done / n, f"识别 {done}/{n}")
            if stream is not None:
                merged = segm.dedup_chunk_segments([list(x) for x in per_chunk[:done]], bounds[:done])
                stream.write_segments(merged[written:])
                written = len(merged)
        return segm.dedup_chunk_segments(per_chunk, bounds)

    def _language(self) -> str | None:
        lang = self.language_override
        if lang is not None:
            return None if lang == "auto" else lang
        eng_cfg = getattr(self.engine, "cfg", None)
        lang = getattr(eng_cfg, "language", None) or "auto"
        return None if lang == "auto" else lang

    def _diarize(self, result: TranscriptionResult, wav_path: Path, samples, step, cancel):
        cfg = self.cfg.diarization
        step("diarize", 0.0, "说话人分离")
        try:
            if self.diarizer_provider is None:
                raise RuntimeError("没有可用的说话人分离后端")
            diarizer = self.diarizer_provider(self.cfg)
            dev = getattr(diarizer, "device", "cpu")
            req = DiarizeRequest(str(wav_path), samples, cfg.num_speakers, cfg.min_speakers, cfg.max_speakers, dev)
            out = diarizer.diarize(req, lambda f, m: step("diarize", f, m), cancel)
            out = normalize_labels(out)
            exclusive = dassign.clean_turns(out.exclusive_turns, cfg.min_turn_duration, cfg.merge_gap_threshold)
            if not exclusive:
                result.warnings.append("说话人分离没有检测到任何说话人")
                return None
            segs, warns = dassign.assign_speakers(result.segments, exclusive)
            result.segments = segs
            result.warnings.extend(warns)
            result.overlaps = dassign.overlaps_from_turns(out.turns)
            for label in sorted({s.speaker for s in segs if s.speaker}):
                result.speakers[label] = SpeakerInfo(label=label)
            step("diarize", 1.0, f"检测到 {len(result.speakers)} 位说话人")
            return out
        except CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - diarization failure must not fail the task
            log.exception("说话人分离失败")
            msg = f"说话人分离失败，已跳过：{e}"
            result.warnings.append(msg)
            self._info(msg)
            return None

    def _voiceprints(self, result: TranscriptionResult, diar, samples, step, cancel) -> None:
        from asr_dashboard.core.voiceprints import matcher

        step("voiceprint", 0.0, "声纹匹配")
        try:
            if self.embedder_provider is None:
                raise RuntimeError("声纹模型不可用")
            embedder = self.embedder_provider(self.cfg)
            same_model = (embedder.kind == "pyannote" and diar.backend == "pyannote" and diar.embeddings)
            vecs: dict[str, np.ndarray] = {}
            labels = list(result.speakers)
            for i, label in enumerate(labels):
                cancel.raise_if_cancelled()
                if same_model and label in diar.embeddings:
                    v = diar.embeddings[label]
                    vecs[label] = v / max(np.linalg.norm(v), 1e-9)
                else:
                    regions = dassign.speaker_clean_regions(label, result.segments, result.overlaps, target=30)
                    if not regions:
                        continue
                    clip = np.concatenate([audiomod.slice_samples(samples, a, b) for a, b in regions])
                    vecs[label] = embedder.embed(clip)
                result.speakers[label].embedding = vecs[label].astype(float).tolist()
                step("voiceprint", (i + 1) / len(labels) * 0.8, f"提取声纹 {i + 1}/{len(labels)}")
            store = self.voiceprint_store
            people = store.centroids(embedder.model_id) if store else []
            if store and not people and store.list_people():
                result.warnings.append("声纹库中的样本还没有用当前声纹模型计算向量，请在声纹库页面点击“重新计算”")
            matches = matcher.match_speakers(vecs, people, self.cfg.voiceprint.match_threshold)
            best = matcher.best_scores(vecs, people)
            for label, info in result.speakers.items():
                if label in matches:
                    m = matches[label]
                    info.display_name, info.voiceprint_id, info.match_score = m.name, m.person_id, m.score
                    store.touch_matched(m.person_id)
                elif label in best:
                    info.match_score = best[label]
            step("voiceprint", 1.0, f"匹配到 {len(matches)} 人")
        except CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("声纹匹配失败")
            msg = f"声纹匹配失败，已跳过：{e}"
            result.warnings.append(msg)
            self._info(msg)


def _chunk_index(seg: Segment, part: list[AudioChunk], base: int) -> int:
    mid = (seg.start + seg.end) / 2
    for k, c in enumerate(part):
        if c.offset - 1e-6 <= mid < c.end + 1e-6:
            return base + k
    best = min(range(len(part)), key=lambda k: abs(part[k].offset - seg.start))
    return base + best


def _majority_lang(segments: list[Segment]) -> str | None:
    counts: dict[str, float] = {}
    for s in segments:
        if s.language:
            counts[s.language] = counts.get(s.language, 0.0) + s.duration
    return max(counts, key=counts.get) if counts else None

"""Qt workers: single-file transcription and folder batches (one worker thread each)."""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from asr_dashboard.config.schema import AppConfig
from asr_dashboard.core import batch as batchmod
from asr_dashboard.core.pipeline import PipelineOutput, TaskRequest
from asr_dashboard.models import CancelledError, CancelToken
from asr_dashboard.redact import redact

log = logging.getLogger(__name__)


class TranscriptionWorker(QThread):
    progress = pyqtSignal(str, float, str)  # stage, overall, detail
    info = pyqtSignal(str)
    succeeded = pyqtSignal(object)  # PipelineOutput
    failed = pyqtSignal(str, str)  # message, traceback
    cancelled = pyqtSignal()

    def __init__(self, services, engine_id: str, cfg: AppConfig, req: TaskRequest, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.engine_id = engine_id
        self.cfg = cfg
        self.req = req
        self.token = CancelToken()
        self.started_at = 0.0

    def cancel(self) -> None:
        self.token.cancel()

    def run(self) -> None:
        self.started_at = time.time()
        try:
            pipe = self.services.pipeline(self.engine_id, self.cfg)
            pipe.on_info = self.info.emit
            out: PipelineOutput = pipe.run(self.req, lambda s, o, d: self.progress.emit(s, o, d), self.token)
            entry = self.services.history.add(self.req.source, self.engine_id, "done", time.time() - self.started_at,
                                              out.result, out.outputs)
            out.result.engine_settings.setdefault("history_id", entry.id)
            self.succeeded.emit(out)
        except CancelledError:
            self.services.history.add(self.req.source, self.engine_id, "cancelled", time.time() - self.started_at)
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            tb = redact(traceback.format_exc())
            log.error("转录失败：%s", redact(str(e)))
            self.services.history.add(self.req.source, self.engine_id, "failed", time.time() - self.started_at,
                                      error=redact(str(e)))
            self.failed.emit(redact(str(e)), tb)


class BatchWorker(QThread):
    item_started = pyqtSignal(int)
    item_done = pyqtSignal(int)
    progress = pyqtSignal(int, float, str, float, object)  # idx, file frac, msg, overall, eta seconds|None
    info = pyqtSignal(str)
    paused = pyqtSignal()
    finished_batch = pyqtSignal(object)  # manifest
    failed = pyqtSignal(str, str)

    def __init__(self, services, engine_id: str, cfg: AppConfig, manifest: batchmod.BatchManifest,
                 task_context: str = "", glossary_ids: list[str] | None = None, language: str | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.engine_id = engine_id
        self.cfg = cfg
        self.manifest = manifest
        self.task_context = task_context
        self.glossary_ids = glossary_ids or []
        self.language = language
        self.runner: batchmod.BatchRunner | None = None

    def pause(self) -> None:
        if self.runner:
            self.runner.pause()

    def resume(self) -> None:
        if self.runner:
            self.runner.resume()

    def cancel(self) -> None:
        if self.runner:
            self.runner.stop()

    def run(self) -> None:
        try:
            # models stay loaded for the whole batch
            pipe = self.services.pipeline(self.engine_id, self.cfg, keep_loaded=True)
            pipe.on_info = self.info.emit
            engine = pipe.engine

            def process(item, out_dir: Path, on_existing: str, prog, cancel: CancelToken):
                req = TaskRequest(item.path, self.task_context, self.glossary_ids, str(out_dir), Path(item.rel).stem,
                                  on_existing, self.manifest.batch_id, self.language)
                t0 = time.time()
                out = pipe.run(req, lambda s, o, d: prog(o, d), cancel)
                entry = self.services.history.add(item.path, self.engine_id, "done", time.time() - t0, out.result,
                                                  out.outputs, batch_id=self.manifest.batch_id)
                names = [out.result.speaker_name(k) or k for k in out.result.speakers]
                return batchmod.ProcessOutcome({k: str(v) for k, v in out.outputs.items()}, out.result.duration,
                                               names, len(out.result.warnings), entry.result_path)

            cb = batchmod.BatchCallbacks(
                on_item_start=lambda i, it: self.item_started.emit(i),
                on_item_done=lambda i, it: self.item_done.emit(i),
                on_progress=lambda i, f, m, o, eta: self.progress.emit(i, f, m, o, eta),
                on_info=self.info.emit, on_paused=self.paused.emit,
            )
            self.runner = batchmod.BatchRunner(self.manifest, process, self.cfg.output.on_existing,
                                               self.cfg.batch.prevent_sleep, self.cfg.batch.oom_retry,
                                               engine.reduce_batch_size, cb)
            self.runner.run()
            if not getattr(engine, "cfg", None) or not engine.cfg.keep_loaded:  # type: ignore[attr-defined]
                engine.unload()
            self.finished_batch.emit(self.manifest)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(redact(str(e)), redact(traceback.format_exc()))

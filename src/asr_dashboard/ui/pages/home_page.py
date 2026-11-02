"""Home page: single file transcription."""

from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    StrongBodyLabel,
)
from qfluentwidgets import FluentIcon as FIF

from asr_dashboard.core import audio as audiomod
from asr_dashboard.core.pipeline import PipelineOutput, TaskRequest
from asr_dashboard.core.tasks import TranscriptionWorker
from asr_dashboard.ui.widgets.common import (
    Card,
    Page,
    error,
    fmt_duration,
    info,
    row,
    run_bg,
    success,
    warning,
)
from asr_dashboard.ui.widgets.error_dialog import show_error
from asr_dashboard.ui.widgets.task_options import TaskOptions

EXT_FILTER = "媒体文件 (" + " ".join(f"*{e}" for e in sorted(audiomod.MEDIA_EXTS)) + ");;所有文件 (*)"


class HomePage(Page):
    result_ready = pyqtSignal(object)  # PipelineOutput

    def __init__(self, services, parent=None) -> None:
        super().__init__("homePage", "单文件转录", "选择或拖放一个音频/视频文件，得到带时间戳和说话人的转录文本。")
        self.services = services
        self.file: str | None = None
        self.worker: TranscriptionWorker | None = None
        self.setAcceptDrops(True)

        # --- file
        fc = Card("输入文件")
        self.pick = PrimaryPushButton(FIF.FOLDER_ADD, "选择文件")
        self.pick.clicked.connect(self.choose)
        self.file_label = StrongBodyLabel("未选择文件（也可以直接把文件拖到窗口里）")
        self.file_info = CaptionLabel("")
        fc.lay.addLayout(row(self.pick, self.file_label, None))
        fc.lay.addWidget(self.file_info)
        self.body.addWidget(fc)

        self.options = TaskOptions(services)
        self.body.addWidget(self.options)

        # --- run
        rc = Card("运行")
        self.start_btn = PrimaryPushButton(FIF.PLAY, "开始转录")
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = PushButton(FIF.CLOSE, "取消")
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.setEnabled(False)
        rc.lay.addLayout(row(self.start_btn, self.cancel_btn, None))
        self.bar = ProgressBar()
        self.bar.setRange(0, 1000)
        rc.lay.addWidget(self.bar)
        self.stage = BodyLabel("就绪")
        self.times = CaptionLabel("")
        rc.lay.addLayout(row(self.stage, None, self.times))
        self.body.addWidget(rc)
        self.body.addStretch(1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._overall = 0.0
        self._t0 = 0.0

    # ------------------------------------------------------------------ input
    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p and Path(p).is_file():
                self.set_file(p)
                break

    def choose(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "选择音频或视频文件", "", EXT_FILTER)
        if p:
            self.set_file(p)

    def set_file(self, p: str) -> None:
        if not audiomod.is_media_file(p):
            warning(self, "不支持的文件格式", Path(p).suffix)
        self.file = p
        self.file_label.setText(Path(p).name)
        self.file_info.setText("正在读取文件信息…")

        def done(mi):
            self.file_info.setText(f"时长 {fmt_duration(mi.duration)} · 大小 {mi.size_mb:.1f} MB · 格式 {mi.format}"
                                   + ("" if mi.has_audio else " · ⚠ 未检测到音轨"))

        run_bg(audiomod.probe, p, on_ok=done, on_err=lambda m, tb: self.file_info.setText(f"无法读取：{m}"), parent=self)

    # ------------------------------------------------------------------ run
    def start(self) -> None:
        if not self.file:
            warning(self, "请先选择文件")
            return
        if not audiomod.ffmpeg_path():
            show_error(self, "缺少 FFmpeg", audiomod.FFMPEG_HELP)
            return
        cfg = self.options.build_config()
        self.options.remember()
        req = TaskRequest(self.file, self.options.context.toPlainText(), self.options.selected_glossaries(),
                          language=self.options.language_code())
        self.worker = TranscriptionWorker(self.services, self.options.engine_id(), cfg, req, self)
        self.worker.progress.connect(self._progress)
        self.worker.info.connect(lambda m: warning(self, "提示", m, 8000))
        self.worker.succeeded.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.finished.connect(self._finished)
        self._t0 = time.time()
        self._overall = 0.0
        self.bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.stage.setText("准备中…")
        self.timer.start(500)
        self.worker.start()

    def cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.stage.setText("正在取消…")

    def _progress(self, stage: str, overall: float, detail: str) -> None:
        self._overall = overall
        self.bar.setValue(int(overall * 1000))
        self.stage.setText(detail)

    def _tick(self) -> None:
        el = time.time() - self._t0
        eta = el / self._overall * (1 - self._overall) if self._overall > 0.03 else None
        self.times.setText(f"已用 {fmt_duration(el)} · 预计剩余 {fmt_duration(eta)}")

    def _done(self, out: PipelineOutput) -> None:
        r = out.result
        success(self, "转录完成", f"{len(r.segments)} 段，{len(r.speakers)} 位说话人" +
                (f"，已保存 {len(out.outputs)} 个文件" if out.outputs else ""))
        for w in r.warnings[:3]:
            warning(self, "注意", w, 8000)
        self.result_ready.emit(out)

    def _failed(self, msg: str, tb: str) -> None:
        error(self, "转录失败", msg[:200], 6000)
        show_error(self, "转录失败", msg, tb)

    def _cancelled(self) -> None:
        info(self, "已取消")

    def _finished(self) -> None:
        self.timer.stop()
        self._tick()
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if self.stage.text().startswith("正在取消"):
            self.stage.setText("已取消")

    def refresh(self) -> None:
        self.options.reload()

    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()



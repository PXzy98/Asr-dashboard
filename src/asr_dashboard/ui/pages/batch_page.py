"""Folder batch processing: scan + preview (drag to reorder), strict sequential run, resume, reports."""

from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QAbstractItemView, QFileDialog, QHeaderView, QListWidgetItem, QTableWidgetItem
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    LineEdit,
    ListWidget,
    MessageBox,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    SwitchButton,
    TableWidget,
)
from qfluentwidgets import FluentIcon as FIF

from asr_dashboard.core import batch as B
from asr_dashboard.core.audio import MEDIA_EXTS, ffmpeg_path
from asr_dashboard.core.tasks import BatchWorker
from asr_dashboard.ui.widgets.common import (
    Card,
    Page,
    error,
    fmt_duration,
    info,
    row,
    success,
    warning,
    zh_switch,
)
from asr_dashboard.ui.widgets.error_dialog import show_error
from asr_dashboard.ui.widgets.task_options import TaskOptions


class BatchPage(Page):
    def __init__(self, services, home_options: TaskOptions | None = None, parent=None) -> None:
        super().__init__("batchPage", "批量处理", "严格按顺序逐个处理文件夹中的文件；中断后可以从断点继续。")
        self.services = services
        self.home_options = home_options
        self.folder: str | None = None
        self.manifest: B.BatchManifest | None = None
        self.worker: BatchWorker | None = None

        cfg = services.cfg
        src = Card("1. 选择文件夹并预览清单")
        self.folder_btn = PrimaryPushButton(FIF.FOLDER, "选择文件夹")
        self.folder_btn.clicked.connect(self.choose_folder)
        self.folder_label = BodyLabel("未选择")
        src.lay.addLayout(row(self.folder_btn, self.folder_label, None))
        self.recursive = zh_switch(SwitchButton())
        self.recursive.setChecked(cfg.batch.recursive)
        self.skip_hidden = zh_switch(SwitchButton())
        self.skip_hidden.setChecked(cfg.batch.skip_hidden)
        self.sort = ComboBox()
        for v, t in [("natural", "自然排序"), ("mtime", "修改时间"), ("size", "文件大小")]:
            self.sort.addItem(t, userData=v)
        self.sort.setCurrentIndex(["natural", "mtime", "size"].index(cfg.batch.sort))
        self.exts = LineEdit()
        self.exts.setPlaceholderText("扩展名过滤，如 mp3,wav（留空=全部支持的格式）")
        self.exts.setText(",".join(cfg.batch.extensions))
        rescan = PushButton(FIF.SYNC, "重新扫描")
        rescan.clicked.connect(self.scan)
        src.lay.addLayout(row(BodyLabel("包含子文件夹"), self.recursive, BodyLabel("跳过隐藏/空文件"), self.skip_hidden,
                              BodyLabel("排序"), self.sort, self.exts, rescan))
        self.preview = ListWidget()
        self.preview.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.preview.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.preview.setMinimumHeight(220)
        src.lay.addWidget(self.preview)
        rm = PushButton(FIF.REMOVE, "移除选中")
        rm.clicked.connect(lambda: [self.preview.takeItem(self.preview.row(i)) for i in self.preview.selectedItems()])
        add = PushButton(FIF.ADD, "追加文件")
        add.clicked.connect(self.add_files)
        self.count_label = CaptionLabel("")
        src.lay.addLayout(row(rm, add, None, self.count_label))
        self.body.addWidget(src)

        self.options = TaskOptions(services)
        same = PushButton(FIF.COPY, "沿用首页设置")
        same.clicked.connect(lambda: home_options and self.options.copy_from(home_options))
        self.body.addLayout(row(BodyLabel("2. 处理选项"), None, same))
        self.body.addWidget(self.options)

        run = Card("3. 运行")
        self.out_dir = LineEdit()
        self.out_dir.setPlaceholderText("输出目录（默认：输出目录/文件夹名）")
        self.on_existing = ComboBox()
        for v, t in [("skip", "已有输出：跳过"), ("overwrite", "已有输出：覆盖"), ("rename", "已有输出：另存新文件名")]:
            self.on_existing.addItem(t, userData=v)
        self.on_existing.setCurrentIndex(["skip", "overwrite", "rename"].index(cfg.output.on_existing))
        run.lay.addLayout(row(BodyLabel("输出目录"), self.out_dir, self.on_existing))
        self.start_btn = PrimaryPushButton(FIF.PLAY, "开始")
        self.start_btn.clicked.connect(self.start)
        self.pause_btn = PushButton(FIF.PAUSE, "暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.cancel_btn = PushButton(FIF.CLOSE, "取消")
        self.cancel_btn.clicked.connect(self.cancel)
        self.retry_btn = PushButton(FIF.SYNC, "重试失败项")
        self.retry_btn.clicked.connect(self.retry_failed)
        self.report_btn = PushButton(FIF.DOCUMENT, "打开批次报告")
        self.report_btn.clicked.connect(self.open_report)
        for b in (self.pause_btn, self.cancel_btn, self.retry_btn, self.report_btn):
            b.setEnabled(False)
        run.lay.addLayout(row(self.start_btn, self.pause_btn, self.cancel_btn, self.retry_btn, self.report_btn, None))
        self.file_bar = ProgressBar()
        self.file_bar.setRange(0, 1000)
        self.all_bar = ProgressBar()
        self.all_bar.setRange(0, 1000)
        self.status = BodyLabel("就绪")
        self.eta = CaptionLabel("")
        run.lay.addLayout(row(BodyLabel("当前文件"), self.file_bar))
        run.lay.addLayout(row(BodyLabel("整体进度"), self.all_bar))
        run.lay.addLayout(row(self.status, None, self.eta))
        self.table = TableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["文件", "时长", "状态", "耗时", "信息"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(280)
        run.lay.addWidget(self.table)
        self.body.addWidget(run)
        self._t0 = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ scanning
    def choose_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if d:
            self.folder = d
            self.folder_label.setText(d)
            self.scan()

    def scan(self) -> None:
        if not self.folder:
            return
        exts = [e.strip() for e in self.exts.text().split(",") if e.strip()] or None
        files = B.scan_folder(self.folder, self.recursive.isChecked(), exts, self.skip_hidden.isChecked(),
                              self.skip_hidden.isChecked(), self.sort.currentData())
        self.preview.clear()
        for f in files:
            it = QListWidgetItem(str(Path(f).relative_to(self.folder)))
            it.setData(Qt.ItemDataRole.UserRole, str(f))
            self.preview.addItem(it)
        self.count_label.setText(f"共 {len(files)} 个文件（可拖动调整顺序）")

    def add_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(MEDIA_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "追加文件", self.folder or "", f"媒体文件 ({exts})")
        for f in files:
            it = QListWidgetItem(f)
            it.setData(Qt.ItemDataRole.UserRole, f)
            self.preview.addItem(it)

    def _ordered_files(self) -> list[Path]:
        return [Path(self.preview.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.preview.count())]

    # ------------------------------------------------------------------ running
    def _settings_snapshot(self, cfg) -> dict:
        return {"engine": self.options.engine_id(), "formats": cfg.output.formats, "language": self.options.language_code(),
                "qwen3": cfg.qwen3.model_dump(mode="json"), "segmentation": cfg.segmentation.model_dump(mode="json"),
                "diarization": cfg.diarization.model_dump(mode="json"), "voiceprint": cfg.voiceprint.model_dump(mode="json"),
                "text": cfg.text.model_dump(mode="json"), "llm_active": cfg.llm_active,
                "context": self.options.context.toPlainText(), "glossaries": self.options.selected_glossaries()}

    def start(self) -> None:
        files = self._ordered_files()
        if not files:
            warning(self, "清单为空", "请先选择文件夹")
            return
        if not ffmpeg_path():
            show_error(self, "缺少 FFmpeg", "请先安装 FFmpeg")
            return
        cfg = self.options.build_config()
        cfg.output.on_existing = self.on_existing.currentData()
        out = self.out_dir.text().strip() or str(Path(cfg.app.output_dir) / Path(self.folder or files[0].parent).name)
        self.manifest = B.BatchManifest.create(self.folder or files[0].parent, files, out, self._settings_snapshot(cfg))
        self._run(cfg)

    def resume(self, manifest: B.BatchManifest) -> None:
        """Continue an unfinished batch found at start-up (uses the engine/options stored in the manifest)."""
        self.manifest = manifest
        cfg = self.services.cfg.model_copy(deep=True)
        st = manifest.settings
        try:
            from asr_dashboard.config.schema import AppConfig

            data = cfg.model_dump()
            for k in ("qwen3", "segmentation", "diarization", "voiceprint", "text"):
                if k in st:
                    data[k] = st[k]
            data["output"]["formats"] = st.get("formats", cfg.output.formats)
            cfg = AppConfig.model_validate(data)
        except Exception:  # noqa: BLE001
            pass
        self.folder = manifest.folder
        self.folder_label.setText(manifest.folder)
        self._run(cfg, engine_id=st.get("engine"), context=st.get("context", ""), glossaries=st.get("glossaries", []),
                  language=st.get("language"))

    def _run(self, cfg, engine_id=None, context=None, glossaries=None, language=None) -> None:
        m = self.manifest
        self._fill_table()
        self.worker = BatchWorker(self.services, engine_id or self.options.engine_id(), cfg, m,
                                  self.options.context.toPlainText() if context is None else context,
                                  self.options.selected_glossaries() if glossaries is None else glossaries,
                                  language or self.options.language_code(), self)
        self.worker.item_started.connect(self._item_started)
        self.worker.item_done.connect(self._item_done)
        self.worker.progress.connect(self._progress)
        self.worker.info.connect(lambda msg: warning(self, "提示", msg))
        self.worker.paused.connect(lambda: self.status.setText("已暂停（当前文件已完成）"))
        self.worker.finished_batch.connect(self._finished)
        self.worker.failed.connect(lambda msg, tb: show_error(self, "批量处理出错", msg, tb))
        self.worker.finished.connect(self._worker_stopped)
        self.start_btn.setEnabled(False)
        for b in (self.pause_btn, self.cancel_btn):
            b.setEnabled(True)
        self.retry_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self._t0 = time.time()
        self.timer.start(1000)
        self.worker.start()

    def toggle_pause(self) -> None:
        if not self.worker or not self.worker.runner:
            return
        if self.worker.runner.paused:
            self.worker.resume()
            self.pause_btn.setText("暂停")
            self.status.setText("继续处理")
        else:
            self.worker.pause()
            self.pause_btn.setText("继续")
            self.status.setText("将在当前文件完成后暂停…")

    def cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.status.setText("正在取消…")

    def retry_failed(self) -> None:
        if not self.manifest:
            return
        for i in self.manifest.items:
            if i.status in (B.FAILED, B.CANCELLED):
                i.status, i.error = B.PENDING, ""
        self.manifest.finished_at = None
        self._run(self.options.build_config())

    def open_report(self) -> None:
        if self.manifest:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.manifest.output_dir) / "batch_report.csv")))

    # ------------------------------------------------------------------ table / progress
    def _fill_table(self) -> None:
        m = self.manifest
        self.table.setRowCount(len(m.items))
        for r, it in enumerate(m.items):
            self._set_row(r, it)

    def _set_row(self, r: int, it: B.BatchItem) -> None:
        vals = [it.rel, fmt_duration(it.duration) if it.duration else "", B.STATUS_LABEL.get(it.status, it.status),
                f"{it.elapsed:.1f}s" if it.elapsed else "", it.error or ("、".join(it.speakers) if it.speakers else "")]
        for c, v in enumerate(vals):
            self.table.setItem(r, c, QTableWidgetItem(v))

    def _item_started(self, idx: int) -> None:
        self._set_row(idx, self.manifest.items[idx])
        self.table.scrollToItem(self.table.item(idx, 0))
        self.file_bar.setValue(0)

    def _item_done(self, idx: int) -> None:
        self._set_row(idx, self.manifest.items[idx])

    def _progress(self, idx: int, frac: float, msg: str, overall: float, eta) -> None:
        self.file_bar.setValue(int(frac * 1000))
        self.all_bar.setValue(int(overall * 1000))
        it = self.manifest.items[idx]
        self.status.setText(f"[{idx + 1}/{len(self.manifest.items)}] {it.rel}：{msg}")
        self._eta = eta

    def _tick(self) -> None:
        eta = getattr(self, "_eta", None)
        self.eta.setText(f"已用 {fmt_duration(time.time() - self._t0)} · 预计剩余 {fmt_duration(eta)}")

    def _finished(self, m: B.BatchManifest) -> None:
        c = m.counts()
        total = sum(i.elapsed for i in m.items)
        text = (f"成功 {c.get(B.DONE, 0)} 个，失败 {c.get(B.FAILED, 0)} 个，跳过 {c.get(B.SKIPPED, 0)} 个"
                + (f"，已取消 {c.get(B.CANCELLED, 0)} 个" if c.get(B.CANCELLED) else "") + f"\n总耗时 {fmt_duration(total)}")
        MessageBox("批次结束", text, self.window()).exec()
        if c.get(B.FAILED):
            error(self, "部分文件失败", "可点击“重试失败项”", 6000)
        else:
            success(self, "批次完成")

    def _worker_stopped(self) -> None:
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.report_btn.setEnabled(self.manifest is not None)
        self.retry_btn.setEnabled(bool(self.manifest and any(i.status in (B.FAILED, B.CANCELLED) for i in self.manifest.items)))
        if self.manifest and self.manifest.unfinished:
            info(self, "批次未完成", "已保存进度，下次启动可从断点继续")

    def refresh(self) -> None:
        self.options.reload()

    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()



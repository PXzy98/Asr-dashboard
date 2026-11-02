"""History of tasks (``~/.asr-dashboard/history.json``): reopen, re-export, delete."""

from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidgetItem
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import PushButton, TableWidget

from asr_dashboard.export.writers import ExportOptions, save_outputs
from asr_dashboard.ui.widgets.common import Card, Page, row, success, warning

STATUS = {"done": "完成", "failed": "失败", "cancelled": "已取消"}


class HistoryPage(Page):
    open_result = pyqtSignal(object, str)  # TranscriptionResult, history id

    def __init__(self, services, parent=None) -> None:
        super().__init__("historyPage", "历史记录")
        self.services = services
        card = Card()
        self.table = TableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["时间", "文件", "引擎", "状态", "耗时", "批次", "信息"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(520)
        self.table.doubleClicked.connect(lambda _i: self.open_selected())
        btn_open = PushButton(FIF.VIEW, "打开结果")
        btn_open.clicked.connect(self.open_selected)
        btn_exp = PushButton(FIF.SAVE, "重新导出")
        btn_exp.clicked.connect(self.reexport)
        btn_del = PushButton(FIF.DELETE, "删除记录")
        btn_del.clicked.connect(self.delete)
        btn_ref = PushButton(FIF.SYNC, "刷新")
        btn_ref.clicked.connect(self.refresh)
        card.lay.addLayout(row(btn_open, btn_exp, btn_del, None, btn_ref))
        card.lay.addWidget(self.table)
        self.body.addWidget(card)
        self.entries = []
        self.refresh()

    def refresh(self) -> None:
        self.entries = self.services.history.list()
        self.table.setRowCount(len(self.entries))
        for r, e in enumerate(self.entries):
            vals = [time.strftime("%Y-%m-%d %H:%M", time.localtime(e.created_at)), Path(e.file).name, e.engine,
                    STATUS.get(e.status, e.status), f"{e.elapsed:.1f}s", e.batch_id or "", e.error]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))

    def _selected(self):
        r = self.table.currentRow()
        return self.entries[r] if 0 <= r < len(self.entries) else None

    def open_selected(self) -> None:
        e = self._selected()
        if not e:
            return
        res = self.services.history.load_result(e)
        if res is None:
            warning(self, "没有可打开的结果", "失败或已取消的任务没有结果")
            return
        self.open_result.emit(res, e.id)

    def reexport(self) -> None:
        e = self._selected()
        res = self.services.history.load_result(e) if e else None
        if res is None:
            return
        cfg = self.services.cfg
        out = save_outputs(res, cfg.app.output_dir, Path(res.source_file).stem, cfg.output.formats,
                           ExportOptions.from_config(cfg.output), "rename")
        success(self, "已重新导出", "，".join(str(p) for p in out.values()))

    def delete(self) -> None:
        e = self._selected()
        if e:
            self.services.history.delete(e.id)
            self.refresh()

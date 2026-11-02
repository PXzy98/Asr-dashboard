"""Results page: table / plain text / diff views, editing, speakers, playback, export."""

from __future__ import annotations

import difflib
import html
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QListWidgetItem,
    QStackedWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    LineEdit,
    ListWidget,
    PlainTextEdit,
    PrimaryPushButton,
    PushButton,
    RadioButton,
    RoundMenu,
    SegmentedWidget,
    SubtitleLabel,
    TableWidget,
    TextBrowser,
)
from qfluentwidgets import FluentIcon as FIF

from asr_dashboard.core.diarization.assign import speaker_clean_regions
from asr_dashboard.export.writers import ExportOptions, fmt_clock, render
from asr_dashboard.models import TranscriptionResult
from asr_dashboard.text.postprocess import mode_label, undo_all, undo_segment
from asr_dashboard.ui.widgets.common import Card, Page, error, row, run_bg, success, warning


class RenameDialog(QDialog):
    def __init__(self, label: str, current: str, people: list[str], allow_add: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"重命名 {label}")
        lay = QVBoxLayout(self)
        lay.addWidget(BodyLabel(f"为 {label} 设置名字（所有段落同步更新，导出时使用新名字）"))
        self.edit = LineEdit()
        self.edit.setText(current)
        lay.addWidget(self.edit)
        if people:
            self.combo = ComboBox()
            self.combo.addItem("（从声纹库选择）", userData="")
            for p in people:
                self.combo.addItem(p, userData=p)
            self.combo.currentIndexChanged.connect(lambda _i: self.combo.currentData() and self.edit.setText(self.combo.currentData()))
            lay.addWidget(self.combo)
        self.add_sample = CheckBox("同时把这段声音加入此人的声纹样本")
        self.add_sample.setChecked(allow_add)
        lay.addWidget(self.add_sample)
        b = QHBoxLayout()
        ok = PrimaryPushButton("确定")
        ok.clicked.connect(self.accept)
        cancel = PushButton("取消")
        cancel.clicked.connect(self.reject)
        b.addStretch(1)
        b.addWidget(cancel)
        b.addWidget(ok)
        lay.addLayout(b)


class SaveVoiceprintDialog(QDialog):
    def __init__(self, label: str, default_name: str, people: list[tuple[str, str]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("保存到声纹库")
        lay = QVBoxLayout(self)
        lay.addWidget(BodyLabel(f"把 {label} 的声音保存到声纹库（自动挑选最长、不与他人重叠的片段，约 30 秒）。"))
        lay.addWidget(CaptionLabel("声纹数据只保存在本机，不会发送给任何 API。"))
        self.new = RadioButton("新建说话人")
        self.new.setChecked(True)
        self.name = LineEdit()
        self.name.setText(default_name if not default_name.startswith("SPEAKER_") else "")
        self.name.setPlaceholderText("名字")
        self.existing = RadioButton("添加到已有说话人")
        self.combo = ComboBox()
        for pid, name in people:
            self.combo.addItem(name, userData=pid)
        self.existing.setEnabled(bool(people))
        lay.addWidget(self.new)
        lay.addWidget(self.name)
        lay.addWidget(self.existing)
        lay.addWidget(self.combo)
        b = QHBoxLayout()
        ok = PrimaryPushButton("保存")
        ok.clicked.connect(self.accept)
        cancel = PushButton("取消")
        cancel.clicked.connect(self.reject)
        b.addStretch(1)
        b.addWidget(cancel)
        b.addWidget(ok)
        lay.addLayout(b)


class ResultsPage(Page):
    def __init__(self, services, parent=None) -> None:
        super().__init__("resultsPage", "结果", "点击任意一行从对应时间播放原音频；双击文本可编辑；右键说话人可保存到声纹库。")
        self.services = services
        self.result: TranscriptionResult | None = None
        self.history_id: str | None = None
        self._filling = False
        self.player = None

        head = Card()
        self.file_label = SubtitleLabel("暂无结果")
        self.mode_label = CaptionLabel("")
        head.lay.addWidget(self.file_label)
        head.lay.addWidget(self.mode_label)
        self.timings = CaptionLabel("")
        self.timings.setWordWrap(True)
        head.lay.addWidget(self.timings)
        self.warn_list = ListWidget()
        self.warn_list.setFixedHeight(80)
        head.lay.addWidget(self.warn_list)
        self.body.addWidget(head)

        # speakers + suggestions
        spk = Card("说话人", "双击重命名；右键可“保存到声纹库”。")
        self.spk_list = ListWidget()
        self.spk_list.setFixedHeight(120)
        self.spk_list.itemDoubleClicked.connect(self.rename_speaker)
        self.spk_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.spk_list.customContextMenuRequested.connect(self._spk_menu)
        spk.lay.addWidget(self.spk_list)
        self.suggest_box = QVBoxLayout()
        spk.lay.addLayout(self.suggest_box)
        self.body.addWidget(spk)

        # summary
        self.summary_card = Card("摘要与要点")
        self.summary = TextBrowser()
        self.summary.setFixedHeight(160)
        self.summary_card.lay.addWidget(self.summary)
        self.body.addWidget(self.summary_card)
        self.summary_card.hide()

        # views
        view = Card()
        self.seg = SegmentedWidget()
        self.seg.addItem("table", "表格")
        self.seg.addItem("text", "纯文本")
        self.seg.addItem("diff", "显示修改")
        self.seg.setCurrentItem("table")
        self.seg.currentItemChanged.connect(self._switch)
        self.filter = ComboBox()
        self.filter.currentIndexChanged.connect(lambda _i: self.fill_table())
        undo_btn = PushButton(FIF.RETURN, "全部撤销修改")
        undo_btn.clicked.connect(self.undo_all)
        save_btn = PushButton(FIF.SAVE, "保存编辑")
        save_btn.clicked.connect(self.save_history)
        view.lay.addLayout(row(self.seg, BodyLabel("说话人"), self.filter, None, undo_btn, save_btn))
        self.stack = QStackedWidget()
        self.table = TableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["开始", "结束", "说话人", "文本"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setWordWrap(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.table.cellClicked.connect(self._play_row)
        self.table.itemChanged.connect(self._edited)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._row_menu)
        self.table.setMinimumHeight(420)
        self.plain = PlainTextEdit()
        self.plain.setReadOnly(True)
        self.diff = TextBrowser()
        for w in (self.table, self.plain, self.diff):
            self.stack.addWidget(w)
        view.lay.addWidget(self.stack)
        self.body.addWidget(view)

        exp = Card("导出")
        btns = []
        for fmt in ("txt", "srt", "vtt", "json", "ass", "md"):
            b = PushButton(fmt.upper())
            b.clicked.connect(lambda _c=False, f=fmt: self.export(f))
            btns.append(b)
        self.md_btn = btns[-1]
        exp.lay.addLayout(row(*btns, None))
        self.body.addWidget(exp)

    # ------------------------------------------------------------------ load
    def show_result(self, result: TranscriptionResult, history_id: str | None = None) -> None:
        self.result = result
        self.history_id = history_id or result.engine_settings.get("history_id")
        cfg = self.services.cfg
        self.file_label.setText(Path(result.source_file).name)
        self.mode_label.setText(f"引擎：{result.engine_id} · 语言：{result.language or '未知'} · 时长 {fmt_clock(result.duration)}"
                                f" · 后处理：{mode_label(result, cfg)}")
        self.timings.setText("耗时：" + "，".join(f"{k} {v:.1f}s" for k, v in result.timings.items()))
        self.warn_list.clear()
        for w in result.warnings:
            self.warn_list.addItem(QListWidgetItem("⚠ " + w))
        self.warn_list.setVisible(bool(result.warnings))
        self.summary_card.setVisible(bool(result.summary))
        self.summary.setMarkdown(result.summary or "")
        self.md_btn.setEnabled(bool(result.summary))
        self.fill_speakers()
        self.fill_table()
        self._load_media()

    def fill_speakers(self) -> None:
        r = self.result
        self.spk_list.clear()
        self.filter.blockSignals(True)
        self.filter.clear()
        self.filter.addItem("全部", userData="")
        for label, info in sorted(r.speakers.items()):
            score = f"（匹配得分 {info.match_score:.2f}）" if info.match_score is not None else ""
            it = QListWidgetItem(f"{label} → {info.name}{score}" if info.display_name else f"{label}{score}")
            it.setData(Qt.ItemDataRole.UserRole, label)
            self.spk_list.addItem(it)
            self.filter.addItem(info.name, userData=label)
        self.filter.blockSignals(False)
        while self.suggest_box.count():
            it = self.suggest_box.takeAt(0)
            if it.layout():
                while it.layout().count():
                    w = it.layout().takeAt(0).widget()
                    if w:
                        w.deleteLater()
            elif it.widget():
                it.widget().deleteLater()
        for s in r.speaker_suggestions:
            if s["speaker"] not in r.speakers or r.speakers[s["speaker"]].display_name:
                continue
            b = PushButton("采用")
            b.clicked.connect(lambda _c=False, sug=s: self._apply_suggestion(sug))
            self.suggest_box.addLayout(row(CaptionLabel(f"大模型推测：{s['speaker']} 可能是“{s['name']}”（{s['reason']}）"), b, None))

    def fill_table(self) -> None:
        if not self.result:
            return
        flt = self.filter.currentData() or ""
        self._filling = True
        self.table.setRowCount(0)
        self._rows: list[int] = []
        for i, s in enumerate(self.result.segments):
            if flt and s.speaker != flt:
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, text in enumerate([fmt_ts(s.start), fmt_ts(s.end), self.result.speaker_name(s.speaker) or "", s.text]):
                it = QTableWidgetItem(text)
                if c < 3:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if c == 3 and s.original_text is not None:
                    it.setToolTip(f"原文：{s.original_text}")
                self.table.setItem(r, c, it)
            self._rows.append(i)
        self.table.resizeRowsToContents()
        self._filling = False
        self.plain.setPlainText(render(self.result, "txt", ExportOptions(txt_timestamps=True)) or "")
        self.diff.setHtml(self._diff_html())

    def _diff_html(self) -> str:
        parts = ["<style>del{background:#fdd;color:#900}ins{background:#dfd;color:#060;text-decoration:none}</style>"]
        changed = 0
        for i, s in enumerate(self.result.segments):
            if s.original_text is None or s.original_text == s.text:
                continue
            changed += 1
            sm = difflib.SequenceMatcher(None, s.original_text, s.text)
            out = []
            for op, a0, a1, b0, b1 in sm.get_opcodes():
                if op == "equal":
                    out.append(html.escape(s.text[b0:b1]))
                if op in ("delete", "replace"):
                    out.append(f"<del>{html.escape(s.original_text[a0:a1])}</del>")
                if op in ("insert", "replace"):
                    out.append(f"<ins>{html.escape(s.text[b0:b1])}</ins>")
            parts.append(f"<p><b>#{i + 1} [{fmt_ts(s.start)}]</b> {''.join(out)}</p>")
        if not changed:
            parts.append("<p>没有被后处理修改过的段落。</p>")
        return "".join(parts)

    def _switch(self, key: str) -> None:
        self.stack.setCurrentIndex({"table": 0, "text": 1, "diff": 2}[key])

    # ------------------------------------------------------------------ editing
    def _edited(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() != 3 or not self.result:
            return
        idx = self._rows[item.row()]
        self.result.segments[idx].text = item.text()
        self.result.rebuild_text()

    def _row_menu(self, pos) -> None:
        r = self.table.rowAt(pos.y())
        if r < 0 or not self.result:
            return
        idx = self._rows[r]
        menu = RoundMenu(parent=self)
        a = Action(FIF.RETURN, "撤销此条修改")
        a.setEnabled(self.result.segments[idx].original_text is not None)
        a.triggered.connect(lambda: (undo_segment(self.result, idx), self.fill_table()))
        menu.addAction(a)
        spk = self.result.segments[idx].speaker
        if spk:
            b = Action(FIF.PEOPLE, f"把 {self.result.speaker_name(spk)} 保存到声纹库")
            b.triggered.connect(lambda: self.save_voiceprint(spk))
            menu.addAction(b)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def undo_all(self) -> None:
        if self.result:
            n = undo_all(self.result)
            self.fill_table()
            success(self, f"已撤销 {n} 条修改")

    def save_history(self) -> None:
        if self.result and self.history_id:
            self.services.history.update_result(self.history_id, self.result)
            success(self, "已保存编辑")

    # ------------------------------------------------------------------ speakers
    def _spk_menu(self, pos) -> None:
        it = self.spk_list.itemAt(pos)
        if not it:
            return
        label = it.data(Qt.ItemDataRole.UserRole)
        menu = RoundMenu(parent=self)
        a = Action(FIF.EDIT, "重命名")
        a.triggered.connect(lambda: self.rename_speaker(it))
        b = Action(FIF.PEOPLE, "保存到声纹库")
        b.triggered.connect(lambda: self.save_voiceprint(label))
        menu.addActions([a, b])
        menu.exec(self.spk_list.viewport().mapToGlobal(pos))

    def rename_speaker(self, item) -> None:
        label = item.data(Qt.ItemDataRole.UserRole)
        info = self.result.speakers[label]
        people = [p.name for p in self.services.voiceprints.list_people()]
        dlg = RenameDialog(label, info.name, people, self.services.cfg.voiceprint.auto_add_on_correction, self)
        if not dlg.exec():
            return
        name = dlg.edit.text().strip()
        info.display_name = name or None
        self.fill_speakers()
        self.fill_table()
        if name and dlg.add_sample.isChecked():
            existing = self.services.voiceprints.find_by_name(name)
            self._store_voice(label, name, existing.id if existing else None)

    def _apply_suggestion(self, sug: dict) -> None:
        self.result.speakers[sug["speaker"]].display_name = sug["name"]
        self.fill_speakers()
        self.fill_table()

    def save_voiceprint(self, label: str) -> None:
        people = [(p.id, p.name) for p in self.services.voiceprints.list_people()]
        dlg = SaveVoiceprintDialog(label, self.result.speakers[label].name, people, self)
        if not dlg.exec():
            return
        if dlg.new.isChecked():
            if not dlg.name.text().strip():
                warning(self, "请输入名字")
                return
            self._store_voice(label, dlg.name.text().strip(), None)
        else:
            self._store_voice(label, dlg.combo.currentText(), dlg.combo.currentData())

    def _store_voice(self, label: str, name: str, person_id: str | None) -> None:
        r = self.result
        cfg = self.services.cfg
        regions = speaker_clean_regions(label, r.segments, r.overlaps, target=30)
        if not regions:
            warning(self, "没有找到足够长的单人片段", "该说话人的段落都太短或与他人重叠")
            return

        def work():
            from asr_dashboard.core import audio as A

            wav = A.to_wav16k(r.source_file, cfg.app.temp_dir or None)
            try:
                samples = A.load_wav(wav)
            finally:
                A.safe_unlink(wav)
            clip = np.concatenate([A.slice_samples(samples, a, b) for a, b in regions])
            store = self.services.voiceprints
            pid = person_id or store.add_person(name).id
            emb = self.services.embedder(cfg)
            vec = emb.embed(clip)
            store.add_sample(pid, clip, f"result:{Path(r.source_file).name}:{label}", vec, emb.model_id)
            return pid, len(clip) / 16000

        def ok(res):
            pid, dur = res
            self.result.speakers[label].display_name = name
            self.result.speakers[label].voiceprint_id = pid
            self.fill_speakers()
            self.fill_table()
            success(self, "已保存到声纹库", f"{name}：{dur:.0f} 秒样本")

        run_bg(work, on_ok=ok, on_err=lambda m, tb: error(self, "保存声纹失败", m), parent=self)

    # ------------------------------------------------------------------ playback / export
    def _load_media(self) -> None:
        try:
            from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
        except Exception:  # noqa: BLE001
            return
        if self.player is None:
            self.player = QMediaPlayer(self)
            self.audio_out = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_out)
        self.player.setSource(QUrl.fromLocalFile(self.result.source_file))

    def _play_row(self, r: int, _c: int) -> None:
        if self.player is None or not self.result:
            return
        s = self.result.segments[self._rows[r]]
        self.player.setPosition(int(s.start * 1000))
        self.player.play()

    def export(self, fmt: str) -> None:
        if not self.result:
            return
        cfg = self.services.cfg
        default = str(Path(cfg.app.output_dir) / f"{Path(self.result.source_file).stem}.{fmt}")
        p, _ = QFileDialog.getSaveFileName(self, f"导出 {fmt.upper()}", default, f"{fmt.upper()} (*.{fmt})")
        if not p:
            return
        content = render(self.result, fmt, ExportOptions.from_config(cfg.output))
        if content is None:
            warning(self, "没有可导出的内容")
            return
        Path(p).write_text(content, encoding="utf-8")
        success(self, "已导出", p)

    def stop_playback(self) -> None:
        if self.player is not None:
            self.player.stop()


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"



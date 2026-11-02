"""Voiceprint library page: people, samples (file / microphone / results), import/export, recompute."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QTableWidgetItem,
    QVBoxLayout,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    DoubleSpinBox,
    LineEdit,
    MessageBox,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    SubtitleLabel,
    TableWidget,
)
from qfluentwidgets import FluentIcon as FIF

from asr_dashboard.core import audio as A
from asr_dashboard.core.voiceprints.embedder import check_sample_quality
from asr_dashboard.ui.widgets.common import Card, Page, error, fmt_duration, row, run_bg, success, warning


class RecorderDialog(QDialog):
    """Microphone recording with a level meter and duration."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("麦克风录音")
        self.frames: list[np.ndarray] = []
        self.stream = None
        lay = QVBoxLayout(self)
        lay.addWidget(BodyLabel("请一个人清晰地说话，建议至少 10 秒。"))
        self.level = ProgressBar()
        self.level.setRange(0, 100)
        self.dur = SubtitleLabel("0.0 秒")
        lay.addWidget(self.level)
        lay.addWidget(self.dur)
        self.rec = PrimaryPushButton(FIF.MICROPHONE, "开始录音")
        self.rec.clicked.connect(self.toggle)
        ok = PushButton("使用这段录音")
        ok.clicked.connect(self.finish)
        lay.addLayout(row(self.rec, ok, None))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update)
        self._level = 0.0

    def toggle(self) -> None:
        if self.stream is None:
            try:
                from asr_dashboard.realtime.capture import MicSource

                self.frames = []
                self.stream = MicSource(None, self._on_frames)
                self.stream.start()
                self.rec.setText("停止录音")
                self.timer.start(100)
            except Exception as e:  # noqa: BLE001
                error(self, "无法打开麦克风", str(e))
                self.stream = None
        else:
            self.stream.stop()
            self.stream = None
            self.rec.setText("重新录音")
            self.timer.stop()

    def _on_frames(self, f: np.ndarray) -> None:
        self.frames.append(f)
        self._level = float(np.sqrt(np.mean(f**2))) if len(f) else 0.0

    def _update(self) -> None:
        self.level.setValue(int(min(1.0, self._level * 8) * 100))
        self.dur.setText(f"{sum(len(f) for f in self.frames) / 16000:.1f} 秒")

    def finish(self) -> None:
        if self.stream is not None:
            self.toggle()
        self.accept()

    def samples(self) -> np.ndarray:
        return np.concatenate(self.frames) if self.frames else np.zeros(0, np.float32)


class AddSampleDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("从文件添加样本")
        self.path = ""
        lay = QVBoxLayout(self)
        pick = PushButton(FIF.FOLDER, "选择音频或视频文件")
        pick.clicked.connect(self.choose)
        self.label = CaptionLabel("")
        self.start = DoubleSpinBox()
        self.start.setRange(0, 1e6)
        self.end = DoubleSpinBox()
        self.end.setRange(0, 1e6)
        lay.addLayout(row(pick, self.label, None))
        lay.addLayout(row(BodyLabel("开始（秒）"), self.start, BodyLabel("结束（秒，0=到结尾）"), self.end))
        ok = PrimaryPushButton("添加")
        ok.clicked.connect(self.accept)
        lay.addLayout(row(None, ok))

    def choose(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if p:
            self.path = p
            self.label.setText(Path(p).name)


class VoiceprintsPage(Page):
    def __init__(self, services, parent=None) -> None:
        super().__init__("voiceprintsPage", "声纹库",
                         "登记说话人的声音和名字，转录时自动把 SPEAKER_xx 换成真实姓名。声纹数据只保存在本机，绝不会发送给任何 API。")
        self.services = services
        top = Card()
        add = PrimaryPushButton(FIF.ADD, "新增说话人")
        add.clicked.connect(self.add_person)
        imp = PushButton(FIF.DOWNLOAD, "导入 zip")
        imp.clicked.connect(self.import_zip)
        exp = PushButton(FIF.SHARE, "导出 zip")
        exp.clicked.connect(self.export_zip)
        recompute = PushButton(FIF.SYNC, "用当前声纹模型重新计算")
        recompute.clicked.connect(self.recompute)
        top.lay.addLayout(row(add, imp, exp, recompute, None))
        self.model_label = CaptionLabel("")
        top.lay.addWidget(self.model_label)
        self.people = TableWidget()
        self.people.setColumnCount(4)
        self.people.setHorizontalHeaderLabels(["名字", "样本数", "总时长", "最近匹配"])
        self.people.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.people.verticalHeader().hide()
        self.people.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.people.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.people.itemSelectionChanged.connect(self.show_person)
        self.people.setMinimumHeight(220)
        top.lay.addWidget(self.people)
        self.body.addWidget(top)

        det = Card("详情")
        self.name = LineEdit()
        self.note = LineEdit()
        save = PushButton(FIF.SAVE, "保存名字和备注")
        save.clicked.connect(self.save_person)
        delete = PushButton(FIF.DELETE, "删除此人")
        delete.clicked.connect(self.delete_person)
        det.lay.addLayout(row(BodyLabel("名字"), self.name, BodyLabel("备注"), self.note, save, delete))
        add_file = PushButton(FIF.FOLDER_ADD, "从文件添加样本")
        add_file.clicked.connect(self.add_from_file)
        add_mic = PushButton(FIF.MICROPHONE, "麦克风录音")
        add_mic.clicked.connect(self.add_from_mic)
        play = PushButton(FIF.PLAY, "试听")
        play.clicked.connect(self.play_sample)
        del_s = PushButton(FIF.REMOVE, "删除样本")
        del_s.clicked.connect(self.delete_sample)
        det.lay.addLayout(row(add_file, add_mic, play, del_s, None))
        det.lay.addWidget(CaptionLabel("也可以在结果页右键某个说话人 → “保存到声纹库”。建议样本为干净的单人说话，至少 10 秒。"))
        self.samples = TableWidget()
        self.samples.setColumnCount(4)
        self.samples.setHorizontalHeaderLabels(["样本", "时长", "来源", "已计算的模型"])
        self.samples.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.samples.verticalHeader().hide()
        self.samples.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.samples.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.samples.setMinimumHeight(180)
        det.lay.addWidget(self.samples)
        self.body.addWidget(det)
        self.player = None
        self._people = []
        self.refresh()

    # ------------------------------------------------------------------ listing
    @property
    def store(self):
        return self.services.voiceprints

    def refresh(self) -> None:
        emb = self.services.embedder(self.services.cfg)
        ok, reason = emb.available()
        self.model_label.setText(f"当前声纹模型：{emb.model_id}" + ("" if ok else f"（不可用：{reason}）"))
        self._people = self.store.list_people()
        self.people.setRowCount(len(self._people))
        for r, p in enumerate(self._people):
            last = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.last_matched_at)) if p.last_matched_at else "—"
            for c, v in enumerate([p.name, str(len(p.samples)), fmt_duration(p.total_duration), last]):
                self.people.setItem(r, c, QTableWidgetItem(v))
        self.show_person()

    def _person(self):
        r = self.people.currentRow()
        return self._people[r] if 0 <= r < len(self._people) else None

    def show_person(self) -> None:
        p = self._person()
        self.name.setText(p.name if p else "")
        self.note.setText(p.note if p else "")
        samples = p.samples if p else []
        self.samples.setRowCount(len(samples))
        for r, s in enumerate(samples):
            for c, v in enumerate([s.file.name, f"{s.duration:.1f}s", s.source, "、".join(s.models)]):
                it = QTableWidgetItem(v)
                it.setData(Qt.ItemDataRole.UserRole, s.id)
                self.samples.setItem(r, c, it)

    # ------------------------------------------------------------------ people
    def add_person(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("新增说话人")
        lay = QVBoxLayout(dlg)
        name = LineEdit()
        name.setPlaceholderText("名字")
        note = LineEdit()
        note.setPlaceholderText("备注（可选）")
        ok = PrimaryPushButton("创建")
        ok.clicked.connect(dlg.accept)
        for w in (name, note):
            lay.addWidget(w)
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(ok)
        lay.addLayout(h)
        if dlg.exec() and name.text().strip():
            self.store.add_person(name.text().strip(), note.text())
            self.refresh()
            self.people.selectRow(next(i for i, p in enumerate(self._people) if p.name == name.text().strip()))
            success(self, "已创建", "接下来请为他/她添加声音样本")

    def save_person(self) -> None:
        p = self._person()
        if p:
            try:
                self.store.update_person(p.id, self.name.text(), self.note.text())
            except ValueError as e:
                warning(self, str(e))
            self.refresh()

    def delete_person(self) -> None:
        p = self._person()
        if p and MessageBox("删除说话人", f"确定删除 {p.name} 及其全部样本？", self.window()).exec():
            self.store.delete_person(p.id)
            self.refresh()

    # ------------------------------------------------------------------ samples
    def _add_samples(self, p, samples: np.ndarray, source: str) -> None:
        warns = check_sample_quality(samples)
        cfg = self.services.cfg
        emb = self.services.embedder(cfg)

        def work():
            vec = None
            ok, _ = emb.available()
            if ok:
                vec = emb.embed(samples)
            multi = _looks_multi_speaker(samples, emb) if ok else None
            self.store.add_sample(p.id, samples, source, vec, emb.model_id if vec is not None else None)
            return multi, ok

        def done(res):
            multi, ok = res
            w = warns + (check_sample_quality(samples, True)[-1:] if multi else [])
            if not ok:
                w.append("声纹模型不可用，样本已保存，模型可用后请点击“重新计算”")
            self.refresh()
            (warning if w else success)(self, "样本已添加", "；".join(w))

        run_bg(work, on_ok=done, on_err=lambda m, tb: error(self, "添加样本失败", m), parent=self)

    def add_from_file(self) -> None:
        p = self._person()
        if not p:
            warning(self, "请先选择一个说话人")
            return
        dlg = AddSampleDialog(self)
        if not dlg.exec() or not dlg.path:
            return
        start, end = dlg.start.value() or None, dlg.end.value() or None
        cfg = self.services.cfg

        def load():
            wav = A.to_wav16k(dlg.path, cfg.app.temp_dir or None, start=start, end=end)
            try:
                return A.load_wav(wav)
            finally:
                A.safe_unlink(wav)

        run_bg(load, on_ok=lambda s: self._add_samples(p, s, f"file:{Path(dlg.path).name}"),
               on_err=lambda m, tb: error(self, "读取音频失败", m), parent=self)

    def add_from_mic(self) -> None:
        p = self._person()
        if not p:
            warning(self, "请先选择一个说话人")
            return
        dlg = RecorderDialog(self)
        if dlg.exec():
            s = dlg.samples()
            if len(s) < 16000:
                warning(self, "录音太短")
                return
            self._add_samples(p, s, "microphone")

    def _sample_id(self):
        r = self.samples.currentRow()
        it = self.samples.item(r, 0) if r >= 0 else None
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def play_sample(self) -> None:
        p = self._person()
        sid = self._sample_id()
        if not p or not sid:
            return
        s = next(x for x in p.samples if x.id == sid)
        try:
            from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

            if self.player is None:
                self.player = QMediaPlayer(self)
                self._ao = QAudioOutput(self)
                self.player.setAudioOutput(self._ao)
            self.player.setSource(QUrl.fromLocalFile(str(s.file)))
            self.player.play()
        except Exception as e:  # noqa: BLE001
            error(self, "无法播放", str(e))

    def delete_sample(self) -> None:
        sid = self._sample_id()
        if sid:
            self.store.delete_sample(sid)
            self.refresh()

    # ------------------------------------------------------------------ library
    def export_zip(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "导出声纹库", "voiceprints.zip", "Zip (*.zip)")
        if p:
            self.store.export_zip(p)
            success(self, "已导出", p)

    def import_zip(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "导入声纹库", "", "Zip (*.zip)")
        if p:
            try:
                n = self.store.import_zip(p)
                self.refresh()
                success(self, "导入完成", f"新增 {n} 人（同名的人合并样本）")
            except Exception as e:  # noqa: BLE001
                error(self, "导入失败", str(e))

    def recompute(self) -> None:
        emb = self.services.embedder(self.services.cfg)
        ok, reason = emb.available()
        if not ok:
            error(self, "声纹模型不可用", reason)
            return
        run_bg(lambda progress: self.store.recompute(emb.model_id, emb.embed, progress),
               on_ok=lambda n: (self.refresh(), success(self, "重新计算完成", f"{n} 个样本")),
               on_err=lambda m, tb: error(self, "重新计算失败", m),
               on_progress=lambda f, m: self.model_label.setText(m), parent=self)


def _looks_multi_speaker(samples: np.ndarray, emb, window: float = 3.0) -> bool:
    """Heuristic: windows far apart in embedding space suggest more than one speaker."""
    n = int(window * 16000)
    if len(samples) < n * 3:
        return False
    vecs = [emb._raw(samples[i:i + n]) for i in range(0, len(samples) - n + 1, n)][:10]
    vecs = [v / max(np.linalg.norm(v), 1e-9) for v in vecs]
    mean = np.mean(vecs, axis=0)
    sims = [float(v @ mean / max(np.linalg.norm(mean), 1e-9)) for v in vecs]
    return min(sims) < 0.5

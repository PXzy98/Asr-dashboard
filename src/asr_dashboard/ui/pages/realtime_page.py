"""Realtime transcription page (requires a validated realtime service in Settings)."""

from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    TextBrowser,
)
from qfluentwidgets import FluentIcon as FIF

from asr_dashboard.config import secrets as sec
from asr_dashboard.core.context import build_context
from asr_dashboard.export.writers import fmt_clock
from asr_dashboard.realtime import capture
from asr_dashboard.realtime.session import RealtimeSession, SessionCallbacks, SessionSettings
from asr_dashboard.ui.widgets.common import Card, Page, error, row, success, warning


class _Bridge(QObject):
    partial = pyqtSignal(str)
    final = pyqtSignal(dict)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    level = pyqtSignal(float)
    speaker = pyqtSignal(int, str)
    stopped = pyqtSignal(dict)


class RealtimePage(Page):
    refine_requested = pyqtSignal(str, str)  # wav path, draft text

    def __init__(self, services, parent=None) -> None:
        super().__init__("realtimePage", "实时转录", "连接外部流式识别服务实时出字；同时录音存盘，停止后可自动精修。")
        self.services = services
        self.session: RealtimeSession | None = None
        self.cap: capture.AudioCapture | None = None
        self.b = _Bridge()
        self.b.partial.connect(self._partial)
        self.b.final.connect(self._final)
        self.b.status.connect(lambda s: self.status.setText(s))
        self.b.error.connect(self._error)
        self.b.level.connect(lambda v: self.level.setValue(int(min(1.0, v * 8) * 100)))
        self.b.speaker.connect(self._speaker)
        self.b.stopped.connect(self._stopped)

        src = Card("音频来源")
        self.source = ComboBox()
        for v, t in [("microphone", "麦克风"), ("system", "系统声音（WASAPI 回环）"), ("mixed", "麦克风 + 系统声音")]:
            self.source.addItem(t, userData=v)
        if not capture.loopback_available():
            self.source.setItemEnabled(1, False)
            self.source.setItemEnabled(2, False)
        self.mic = ComboBox()
        self.sys = ComboBox()
        refresh = PushButton(FIF.SYNC, "刷新设备")
        refresh.clicked.connect(self.load_devices)
        src.lay.addLayout(row(BodyLabel("来源"), self.source, BodyLabel("麦克风"), self.mic, BodyLabel("系统设备"), self.sys,
                              refresh, None))
        self.level = ProgressBar()
        self.level.setRange(0, 100)
        src.lay.addLayout(row(BodyLabel("音量"), self.level))
        self.glossary_checks: dict[str, CheckBox] = {}
        self.gl_row = row(BodyLabel("词库"))
        src.lay.addLayout(self.gl_row)
        self.ctx_note = CaptionLabel("")
        src.lay.addWidget(self.ctx_note)
        self.live_spk = CheckBox("实时说话人标注（本地声纹比对）")
        self.refine = CheckBox("停止后自动精修（按当前首页引擎完整处理一遍）")
        src.lay.addLayout(row(self.live_spk, self.refine, None))
        self.body.addWidget(src)

        ctl = Card()
        self.start_btn = PrimaryPushButton(FIF.MICROPHONE, "开始")
        self.start_btn.clicked.connect(self.start)
        self.pause_btn = PushButton(FIF.PAUSE, "暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn = PushButton(FIF.CLOSE, "停止")
        self.stop_btn.clicked.connect(self.stop)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status = BodyLabel("未开始")
        self.clock = CaptionLabel("")
        ctl.lay.addLayout(row(self.start_btn, self.pause_btn, self.stop_btn, self.status, None, self.clock))
        self.view = TextBrowser()
        self.view.setMinimumHeight(420)
        self.view.verticalScrollBar().valueChanged.connect(self._scrolled)
        ctl.lay.addWidget(self.view)
        self.body.addWidget(ctl)
        self.finals: list[dict] = []
        self.partial_text = ""
        self.autoscroll = True
        self._t0 = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.clock.setText(fmt_clock(time.time() - self._t0)))
        self.refresh()

    # ------------------------------------------------------------------ setup
    def refresh(self) -> None:
        cfg = self.services.cfg
        rt = cfg.realtime
        self.source.setCurrentIndex(["microphone", "system", "mixed"].index(rt.source))
        self.live_spk.setChecked(rt.live_speaker_id)
        self.refine.setChecked(rt.refine_after_stop)
        while self.gl_row.count() > 1:
            w = self.gl_row.takeAt(1).widget()
            if w:
                w.deleteLater()
        self.glossary_checks = {}
        for g in cfg.glossaries:
            cb = CheckBox(g.name)
            cb.setChecked(g.id in rt.glossary_ids)
            self.glossary_checks[g.id] = cb
            self.gl_row.addWidget(cb)
        self.gl_row.addStretch(1)
        self.ctx_note.setText("注意：vLLM 的 /v1/realtime 接口只接受模型名，不接受提示词和语言参数，词库在实时阶段不会生效"
                              "（停止后的精修会使用词库）。" if rt.protocol == "vllm_realtime" else
                              "词库会作为 corpus 发送给百炼实时识别服务。")
        self.load_devices()

    def load_devices(self) -> None:
        self.mic.clear()
        self.mic.addItem("默认麦克风", userData=None)
        for d in capture.list_microphones():
            self.mic.addItem(d.name, userData=d)
        self.sys.clear()
        self.sys.addItem("默认输出设备", userData=None)
        for d in capture.list_loopback():
            self.sys.addItem(d.name, userData=d)

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        cfg = self.services.cfg
        rt = cfg.realtime
        if not cfg.realtime_active:
            warning(self, "实时转录未启用", "请先在 设置 → 实时转录服务 中配置并验证")
            return
        gls = [g for g in cfg.glossaries if self.glossary_checks.get(g.id) and self.glossary_checks[g.id].isChecked()]
        ctx = build_context("", gls, "", cfg.qwen3.context_max_chars).text
        stem = time.strftime("realtime_%Y%m%d_%H%M%S")
        out_dir = Path(cfg.app.output_dir) / "realtime"
        settings = SessionSettings(rt.protocol, rt.url, self.services.secrets.get(sec.REALTIME_KEY), rt.model,
                                   None if rt.language == "auto" else rt.language, rt.chunk_ms, rt.auto_reconnect,
                                   rt.max_reconnects, ctx)
        cb = SessionCallbacks(on_partial=self.b.partial.emit, on_final=self.b.final.emit, on_status=self.b.status.emit,
                              on_error=self.b.error.emit, on_level=self.b.level.emit, on_speaker=self.b.speaker.emit,
                              on_stopped=self.b.stopped.emit)
        live = self._live_identifier() if self.live_spk.isChecked() else None
        self.session = RealtimeSession(settings, out_dir, stem, cb, live)
        self.finals, self.partial_text = [], ""
        self.view.clear()
        try:
            self.session.start()
            self.cap = capture.AudioCapture(self.source.currentData(), self.session.feed, self.mic.currentData(),
                                            self.sys.currentData())
            self.cap.start()
        except Exception as e:  # noqa: BLE001
            if self.session:
                self.session.stop(wait=False)
            error(self, "无法开始实时转录", str(e))
            return
        self._t0 = time.time()
        self.timer.start(500)
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def _live_identifier(self):
        cfg = self.services.cfg
        emb = self.services.embedder(cfg)
        ok, reason = emb.available()
        if not ok:
            warning(self, "无法实时标注说话人", reason)
            return None
        people = self.services.voiceprints.centroids(emb.model_id)
        from asr_dashboard.core.voiceprints import matcher

        def identify(clip):
            v = emb.embed(clip)
            m = matcher.match_speakers({"x": v}, people, cfg.voiceprint.match_threshold)
            return m["x"].name if "x" in m else None

        return identify

    def toggle_pause(self) -> None:
        if not self.session:
            return
        if self.pause_btn.text() == "暂停":
            self.session.pause()
            self.pause_btn.setText("继续")
        else:
            self.session.resume()
            self.pause_btn.setText("暂停")

    def stop(self) -> None:
        if self.cap:
            self.cap.stop()
            self.cap = None
        if self.session:
            self.status.setText("正在结束…")
            self.session.stop(wait=False)
        self.timer.stop()
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------ display
    def _scrolled(self, v: int) -> None:
        bar = self.view.verticalScrollBar()
        self.autoscroll = v >= bar.maximum() - 4

    def _render(self) -> None:
        parts = []
        for s in self.finals:
            spk = f"<b>[{s['speaker']}]</b> " if s.get("speaker") else ""
            color = "#c50" if s.get("marker") else "inherit"
            parts.append(f"<p style='color:{color}'><span style='color:#888'>[{fmt_clock(s['start'])}]</span> {spk}{s['text']}</p>")
        if self.partial_text:
            parts.append(f"<p style='color:#999'>{self.partial_text}</p>")
        bar = self.view.verticalScrollBar()
        pos = bar.value()
        self.view.setHtml("".join(parts))
        if self.autoscroll:
            bar.setValue(bar.maximum())
        else:
            bar.setValue(pos)

    def _partial(self, t: str) -> None:
        self.partial_text = t
        self._render()

    def _final(self, s: dict) -> None:
        self.finals.append(s)
        self._render()

    def _speaker(self, idx: int, name: str) -> None:
        if 0 <= idx < len(self.finals):
            self.finals[idx]["speaker"] = name
            self._render()

    def _error(self, msg: str) -> None:
        error(self, "实时转录出错", msg, 8000)
        self.status.setText(msg)

    def _stopped(self, info: dict) -> None:
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self.stop_btn.setEnabled(False)
        if self.cap:
            self.cap.stop()
            self.cap = None
        self.status.setText("已停止" + (f"（{info['error']}）" if info.get("error") else ""))
        success(self, "录音已保存", info["wav"])
        draft = "\n".join(s["text"] for s in info.get("sentences", []))
        if self.refine.isChecked() and info.get("duration", 0) > 1.0:
            self.refine_requested.emit(info["wav"], draft)
        self.session = None

    def is_busy(self) -> bool:
        return self.session is not None

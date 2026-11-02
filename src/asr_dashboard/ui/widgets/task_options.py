"""Per-task options shared by the single-file page and the batch page."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    PlainTextEdit,
    PrimaryPushButton,
    PushButton,
    SmoothScrollArea,
    SpinBox,
    SubtitleLabel,
    SwitchButton,
    ToolButton,
)
from qfluentwidgets import FluentIcon as FIF

from asr_dashboard.config import secrets as sec
from asr_dashboard.config.schema import QWEN_LANGUAGES, AppConfig
from asr_dashboard.engines import registry
from asr_dashboard.ui.widgets.common import Card, row, zh_switch
from asr_dashboard.ui.widgets.schema_form import SchemaForm

COMMON_API_LANGS = ["zh", "en", "ja", "ko", "yue", "fr", "de", "es", "ru", "pt", "it"]
FORMATS = [("txt", "TXT"), ("srt", "SRT"), ("vtt", "VTT"), ("json", "JSON"), ("ass", "ASS")]


class EngineSettingsDialog(QDialog):
    """Settings form for one engine (auto-generated from its schema)."""

    def __init__(self, services, engine_id: str, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.engine_id = engine_id
        self.setWindowTitle("模型设置")
        self.resize(720, 640)
        cfg = services.cfg
        base, variant = registry.split_id(engine_id)
        lay = QVBoxLayout(self)
        lay.addWidget(SubtitleLabel("本地 Qwen3-ASR 设置" if base == registry.LOCAL_ID else "API 模型设置"))
        fields = registry.settings_schema(engine_id, cfg)
        if base == registry.LOCAL_ID:
            values = cfg.qwen3.model_dump(mode="json")
            if variant:
                values["variant"] = variant
        else:
            m = cfg.api_model(base)
            values = m.model_dump(mode="json") if m else {}
            values["api_key"] = services.secrets.get(sec.api_key_name(base))
        self.form = SchemaForm(fields, values)
        scroll = SmoothScrollArea()
        scroll.setWidget(self.form)
        scroll.setWidgetResizable(True)
        lay.addWidget(scroll, 1)
        self.err = CaptionLabel("")
        self.err.setStyleSheet("color: #d13438")
        lay.addWidget(self.err)
        btns = QHBoxLayout()
        cancel = PushButton("取消")
        cancel.clicked.connect(self.reject)
        ok = PrimaryPushButton("保存")
        ok.clicked.connect(self.save)
        btns.addStretch(1)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        lay.addLayout(btns)

    def save(self) -> None:
        try:
            vals = self.form.values()
        except ValueError as e:
            self.err.setText(str(e))
            return
        base, _ = registry.split_id(self.engine_id)

        def mut(c: AppConfig) -> None:
            if base == registry.LOCAL_ID:
                merged = c.qwen3.model_dump()
                _deep_update(merged, vals)
                c.qwen3 = type(c.qwen3).model_validate(merged)
            else:
                m = c.api_model(base)
                key = vals.pop("api_key", None)
                merged = m.model_dump()
                _deep_update(merged, vals)
                idx = c.api_models.index(m)
                c.api_models[idx] = type(m).model_validate(merged)
                if key is not None:
                    self.services.secrets.set(sec.api_key_name(base), key)

        try:
            self.services.config.update(mut)
        except Exception as e:  # noqa: BLE001
            self.err.setText(f"保存失败：{e}")
            return
        self.accept()


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict) and k not in ("extra",):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


class TaskOptions(QWidget):
    engine_changed = pyqtSignal(str)

    def __init__(self, services, parent=None, show_streaming: bool = True) -> None:
        super().__init__(parent)
        self.services = services
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        # --- engine & language
        card = Card("识别模型")
        self.engine = ComboBox()
        self.engine.setMinimumWidth(320)
        self.engine.currentIndexChanged.connect(self._on_engine)
        gear = ToolButton(FIF.SETTING)
        gear.setToolTip("模型设置")
        gear.clicked.connect(self.open_engine_settings)
        self.language = ComboBox()
        self.language.setMinimumWidth(200)
        card.lay.addLayout(row(BodyLabel("模型"), self.engine, gear, BodyLabel("语言"), self.language, None))
        self.dialect_hint = CaptionLabel("另支持 22 种中文方言，由模型自动识别，无需选择")
        card.lay.addWidget(self.dialect_hint)
        outer.addWidget(card)

        # --- context
        ctx = Card("上下文提示（热词）", "提供专有名词、人名、背景信息，提高这些词的识别准确率。属于“软引导”，不能保证每个词都被识别正确。")
        self.glossary_box = QWidget()
        self.glossary_lay = QHBoxLayout(self.glossary_box)
        self.glossary_lay.setContentsMargins(0, 0, 0, 0)
        ctx.lay.addWidget(BodyLabel("启用词库"))
        ctx.lay.addWidget(self.glossary_box)
        self.context = PlainTextEdit()
        self.context.setPlaceholderText("本次任务的临时上下文，例如：本次会议讨论 Qwen3-ASR 的部署，参会人张三、李四")
        self.context.setFixedHeight(70)
        ctx.lay.addWidget(self.context)
        self.ctx_card = ctx
        outer.addWidget(ctx)

        # --- processing options
        opt = Card("处理选项")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        self.diar = zh_switch(SwitchButton())
        self.diar_backend = ComboBox()
        for v, t in [("auto", "自动"), ("pyannote", "pyannote"), ("diarizen", "DiariZen"), ("embedding", "WavLM 兜底")]:
            self.diar_backend.addItem(t, userData=v)
        self.num_spk = SpinBox()
        self.num_spk.setRange(0, 100)
        self.num_spk.setSpecialValueText("自动")
        self.voiceprint = zh_switch(SwitchButton())
        self.aligner = zh_switch(SwitchButton())
        self.seg = ComboBox()
        for v, t in [("hybrid", "混合（默认）"), ("vad", "VAD"), ("duration", "固定时长")]:
            self.seg.addItem(t, userData=v)
        self.merge = zh_switch(SwitchButton())
        self.fillers = zh_switch(SwitchButton())
        self.streaming = zh_switch(SwitchButton())
        items = [("说话人分离", self.diar), ("后端", self.diar_backend), ("人数", self.num_spk),
                 ("声纹自动识别", self.voiceprint), ("强制对齐（词级时间戳）", self.aligner), ("分段策略", self.seg),
                 ("合并为连贯文本", self.merge), ("去口头禅", self.fillers)]
        if show_streaming:
            items.append(("实时流式写入", self.streaming))
        for i, (label, w) in enumerate(items):
            grid.addWidget(BodyLabel(label), i // 3, (i % 3) * 2)
            grid.addWidget(w, i // 3, (i % 3) * 2 + 1)
        opt.lay.addLayout(grid)
        self.formats: dict[str, CheckBox] = {}
        fmt_row = [BodyLabel("导出格式")]
        for k, t in FORMATS:
            cb = CheckBox(t)
            self.formats[k] = cb
            fmt_row.append(cb)
        fmt_row.append(None)
        opt.lay.addLayout(row(*fmt_row))
        self.pp_mode = CaptionLabel("")
        opt.lay.addWidget(self.pp_mode)
        outer.addWidget(opt)
        self.reload()

    # ------------------------------------------------------------------ populate
    def reload(self) -> None:
        cfg = self.services.cfg
        cur = self.engine.currentData() or _default_engine(cfg)
        self.engine.blockSignals(True)
        self.engine.clear()
        for e in registry.list_engines(cfg):
            prefix = "本地 · " if e.group == "local" else "API · "
            self.engine.addItem(prefix + e.label, userData=e.id)
        idx = max(0, next((i for i in range(self.engine.count()) if self.engine.itemData(i) == cur), 0))
        self.engine.setCurrentIndex(idx)
        self.engine.blockSignals(False)
        # glossaries
        while self.glossary_lay.count():
            it = self.glossary_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.glossary_checks: dict[str, CheckBox] = {}
        for g in cfg.glossaries:
            cb = CheckBox(f"{g.name}（{len(g.terms)}）")
            self.glossary_checks[g.id] = cb
            self.glossary_lay.addWidget(cb)
        if not cfg.glossaries:
            self.glossary_lay.addWidget(CaptionLabel("还没有词库，可在 设置 → 上下文与词库 中添加"))
        self.glossary_lay.addStretch(1)
        d = cfg.diarization
        self.diar.setChecked(d.enabled)
        self.diar_backend.setCurrentIndex(max(0, [self.diar_backend.itemData(i) for i in range(self.diar_backend.count())].index(d.backend)))
        self.num_spk.setValue(d.num_speakers)
        self.voiceprint.setChecked(cfg.voiceprint.enabled)
        self.aligner.setChecked(cfg.qwen3.aligner.enabled)
        self.seg.setCurrentIndex([self.seg.itemData(i) for i in range(self.seg.count())].index(cfg.segmentation.strategy))
        self.merge.setChecked(cfg.text.merge_sentences)
        self.fillers.setChecked(cfg.text.remove_fillers)
        self.streaming.setChecked(cfg.output.streaming)
        for k, cb in self.formats.items():
            cb.setChecked(k in cfg.output.formats)
        self.pp_mode.setText("文本后处理：" + (f"大模型（{cfg.llm.model}）" if cfg.llm_active else "规则"))
        self._on_engine()

    def _on_engine(self, *_a) -> None:
        eid = self.engine.currentData()
        if not eid:
            return
        base, _ = registry.split_id(eid)
        local = base == registry.LOCAL_ID
        cfg = self.services.cfg
        cur_lang = self.language.currentData() or (cfg.qwen3.language if local else "auto")
        self.language.clear()
        self.language.addItem("自动识别", userData="auto")
        if local:
            for c, n in QWEN_LANGUAGES.items():
                self.language.addItem(f"{n} ({c})", userData=c)
        else:
            m = cfg.api_model(base)
            for c in COMMON_API_LANGS:
                self.language.addItem(f"{QWEN_LANGUAGES.get(c, c)} ({c})", userData=c)
            if m and m.language not in ("", "auto"):
                cur_lang = m.language
        idx = next((i for i in range(self.language.count()) if self.language.itemData(i) == cur_lang), 0)
        self.language.setCurrentIndex(idx)
        self.dialect_hint.setVisible(local)
        self.aligner.setEnabled(local)
        self.aligner.setToolTip("" if local else "API 引擎不支持本地强制对齐")
        m = None if local else cfg.api_model(base)
        ctx_ok = local or (m is not None and m.supports_context)
        self.ctx_card.setEnabled(ctx_ok)
        self.ctx_card.setToolTip("" if ctx_ok else "当前 API 模型未开启“支持上下文提示”")
        self.engine_changed.emit(eid)

    def open_engine_settings(self) -> None:
        eid = self.engine.currentData()
        if not eid:
            return
        if EngineSettingsDialog(self.services, eid, self.window()).exec():
            self.reload()

    # ------------------------------------------------------------------ results
    def engine_id(self) -> str:
        return self.engine.currentData()

    def selected_glossaries(self) -> list[str]:
        return [gid for gid, cb in self.glossary_checks.items() if cb.isChecked()]

    def build_config(self) -> AppConfig:
        """Snapshot of the global config with this task's choices applied."""
        cfg = self.services.cfg.model_copy(deep=True)
        cfg.diarization.enabled = self.diar.isChecked()
        cfg.diarization.backend = self.diar_backend.currentData()
        cfg.diarization.num_speakers = self.num_spk.value()
        cfg.voiceprint.enabled = self.voiceprint.isChecked()
        cfg.qwen3.aligner.enabled = self.aligner.isChecked()
        _, variant = registry.split_id(self.engine_id())
        if variant:
            cfg.qwen3.variant = variant
        cfg.segmentation.strategy = self.seg.currentData()
        cfg.text.merge_sentences = self.merge.isChecked()
        cfg.text.remove_fillers = self.fillers.isChecked()
        cfg.output.streaming = self.streaming.isChecked()
        cfg.output.formats = [k for k, cb in self.formats.items() if cb.isChecked()] or ["srt"]
        return cfg

    def language_code(self) -> str:
        return self.language.currentData() or "auto"

    def copy_from(self, other: TaskOptions) -> None:
        self.engine.setCurrentIndex(other.engine.currentIndex())
        self.language.setCurrentIndex(other.language.currentIndex())
        self.context.setPlainText(other.context.toPlainText())
        for gid, cb in self.glossary_checks.items():
            if gid in other.glossary_checks:
                cb.setChecked(other.glossary_checks[gid].isChecked())
        for a, b in [(self.diar, other.diar), (self.voiceprint, other.voiceprint), (self.aligner, other.aligner),
                     (self.merge, other.merge), (self.fillers, other.fillers), (self.streaming, other.streaming)]:
            a.setChecked(b.isChecked())
        self.diar_backend.setCurrentIndex(other.diar_backend.currentIndex())
        self.num_spk.setValue(other.num_spk.value())
        self.seg.setCurrentIndex(other.seg.currentIndex())
        for k, cb in self.formats.items():
            cb.setChecked(other.formats[k].isChecked())

    def remember(self) -> None:
        """Persist the engine choice as default."""
        eid = self.engine_id()
        self.services.config.update(lambda c: setattr(c.engine, "default", eid))


def _default_engine(cfg: AppConfig) -> str:
    d = cfg.engine.default
    if d == registry.LOCAL_ID:
        return f"{registry.LOCAL_ID}:{cfg.qwen3.variant}"
    return d

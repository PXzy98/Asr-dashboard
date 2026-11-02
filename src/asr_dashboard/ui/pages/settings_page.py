"""Settings page: all sections, validated LLM / realtime services, model + environment management."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication, QListWidgetItem
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    HyperlinkButton,
    LineEdit,
    ListWidget,
    MessageBox,
    PasswordLineEdit,
    PlainTextEdit,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    StrongBodyLabel,
    Theme,
    setTheme,
)
from qfluentwidgets import FluentIcon as FIF

from asr_dashboard.config import secrets as sec
from asr_dashboard.config.schema import QWEN_LANGUAGES, AppConfig, Glossary
from asr_dashboard.core.diarization import diarizen_backend, pyannote_backend
from asr_dashboard.engines.qwen3.engine import Qwen3LocalEngine
from asr_dashboard.settings_fields import SettingField as F
from asr_dashboard.ui.widgets.common import Card, Page, error, info, row, run_bg, success, warning
from asr_dashboard.ui.widgets.schema_form import SchemaForm

APP_FIELDS = [
    F("output_dir", "输出目录", "path", "output"),
    F("temp_dir", "临时目录", "path", "", "留空使用系统临时目录"),
    F("cleanup_temp", "任务结束后清理临时文件", "bool", True),
    F("theme", "主题", "enum", "system", choices=[("system", "跟随系统"), ("light", "亮色"), ("dark", "暗色")]),
    F("ui_language", "界面语言", "enum", "zh", "英文界面预留（重启生效）", choices=[("zh", "中文"), ("en", "English（预留）")]),
]
SEG_FIELDS = [
    F("strategy", "分段策略", "enum", "hybrid", choices=[("hybrid", "混合（默认）"), ("vad", "VAD"), ("duration", "固定时长")]),
    F("target_duration", "目标时长（秒）", "float", 60, minimum=5, maximum=1200),
    F("min_segment", "最短（秒）", "float", 5, minimum=0.5, maximum=600),
    F("max_segment", "最长（秒）", "float", 60, "开启强制对齐时自动限制在 180 秒以内", minimum=5, maximum=1200),
    F("overlap", "块间重叠（秒）", "float", 0.5, minimum=0, maximum=10),
    F("silence_threshold_db", "静音阈值（dB）", "float", -40, minimum=-90, maximum=0),
    F("min_silence", "最短静音（秒）", "float", 0.3, minimum=0.05, maximum=5),
    F("search_window", "切点搜索范围（±秒）", "float", 10, minimum=0, maximum=60),
    F("max_sentence_duration", "单句最长（秒）", "float", 15, "按词级时间戳重切句子时使用", minimum=2, maximum=120),
]
DIAR_FIELDS = [
    F("enabled", "默认开启说话人分离", "bool", False),
    F("backend", "后端", "enum", "auto", "auto：pyannote → DiariZen → WavLM 兜底",
      choices=[("auto", "自动"), ("pyannote", "pyannote Community-1"), ("diarizen", "DiariZen"), ("embedding", "WavLM 兜底")]),
    F("num_speakers", "说话人数（0=自动）", "int", 0, minimum=0, maximum=100),
    F("min_speakers", "最少人数", "int", 1, minimum=1, maximum=100),
    F("max_speakers", "最多人数", "int", 20, minimum=1, maximum=100),
    F("device", "设备", "str", "follow", "follow = 跟随 Qwen3 设备；也可填 cpu / cuda:0"),
    F("min_turn_duration", "最短片段（秒）", "float", 0.3, "更短的片段并入相邻说话人", minimum=0, maximum=10),
    F("merge_gap_threshold", "合并间隔（秒）", "float", 0.5, "同一说话人间隔小于此值就合并", minimum=0, maximum=10),
    F("diarizen.model_id", "DiariZen 模型", "str", diarizen_backend.DEFAULT_MODEL),
    F("embedding.model", "兜底声纹模型", "str", "microsoft/wavlm-base-plus-sv", advanced=True),
    F("embedding.clustering", "聚类方式", "enum", "agglomerative",
      choices=[("agglomerative", "层次聚类"), ("spectral", "谱聚类")]),
    F("embedding.similarity_threshold", "相似度阈值", "float", 0.7, minimum=0, maximum=1, step=0.05),
    F("embedding.window", "窗口（秒）", "float", 1.5, minimum=0.5, maximum=10),
    F("embedding.step", "步长（秒）", "float", 0.75, minimum=0.1, maximum=10),
]
VP_FIELDS = [
    F("enabled", "转录后自动识别身份", "bool", True),
    F("embedding_model", "声纹模型", "enum", "pyannote", "库内所有向量必须由同一模型生成；切换后请在声纹库页面重新计算",
      choices=[("pyannote", "pyannote Community-1 自带"), ("wavlm", "WavLM x-vector")]),
    F("match_threshold", "匹配阈值", "float", 0.6, minimum=0, maximum=1, step=0.01),
    F("auto_add_on_correction", "手动更正名字时默认加入声纹样本", "bool", False),
]
TEXT_FIELDS = [
    F("merge_sentences", "合并为连贯文本", "bool", True),
    F("normalize_punctuation", "标点规范化", "bool", True),
    F("clean_whitespace", "空白清理", "bool", True),
    F("smart_capitalize", "英文句首大写", "bool", True),
    F("remove_fillers", "去口头禅", "bool", False),
    F("filler_words", "口头禅词表（JSON）", "json", {}),
]
OUT_FIELDS = [
    F("auto_save", "完成后自动保存", "bool", True),
    F("include_speakers", "导出包含说话人", "bool", True),
    F("txt_timestamps", "TXT 带时间戳", "bool", True),
    F("streaming", "实时流式写入", "bool", False),
    F("ass_font", "ASS 字体", "str", "Arial"),
    F("ass_font_size", "ASS 字号", "int", 20, minimum=6, maximum=200),
    F("on_existing", "已有同名输出", "enum", "skip", "单文件模式下 skip 等同于另存新文件名",
      choices=[("skip", "跳过"), ("overwrite", "覆盖"), ("rename", "另存新文件名")]),
]
BATCH_FIELDS = [
    F("recursive", "包含子文件夹", "bool", True),
    F("sort", "排序", "enum", "natural", choices=[("natural", "自然排序"), ("mtime", "修改时间"), ("size", "文件大小")]),
    F("skip_hidden", "跳过隐藏文件", "bool", True),
    F("skip_empty", "跳过空文件", "bool", True),
    F("prevent_sleep", "处理期间阻止系统睡眠", "bool", True),
    F("oom_retry", "显存不足时批大小减半重试", "bool", True),
]
LOG_FIELDS = [
    F("level", "日志级别", "enum", "INFO", choices=[(x, x) for x in ("DEBUG", "INFO", "WARNING", "ERROR")]),
    F("log_dir", "日志目录", "path", "", "留空：~/.asr-dashboard/logs"),
    F("max_file_size_mb", "单个日志大小（MB）", "int", 10, minimum=1, maximum=1000),
    F("backup_count", "保留份数", "int", 5, minimum=0, maximum=100),
]
LLM_ADV = [
    F("timeout", "超时（秒）", "float", 120, minimum=5, maximum=1800),
    F("temperature", "温度", "float", 0.2, minimum=0, maximum=2, step=0.1),
    F("chunk_chars", "每块最大字数", "int", 1500, minimum=200, maximum=20000),
    F("concurrency", "并发数", "int", 2, minimum=1, maximum=16),
    F("max_change_ratio", "单条最大改动比例", "float", 0.4, minimum=0, maximum=1, step=0.05),
]
RT_ADV = [
    F("chunk_ms", "每次发送的音频块（毫秒）", "int", 100, minimum=20, maximum=2000),
    F("auto_reconnect", "断线自动重连", "bool", True),
    F("max_reconnects", "重连最多次数", "int", 5, minimum=0, maximum=100),
    F("refine_after_stop", "结束后自动精修", "bool", True),
    F("live_speaker_id", "实时说话人标注", "bool", False),
    F("source", "默认音频来源", "enum", "microphone",
      choices=[("microphone", "麦克风"), ("system", "系统声音"), ("mixed", "混合")]),
]


class SettingsPage(Page):
    def __init__(self, services, on_changed: Callable[[], None] | None = None, open_voiceprints=None,
                 open_api_models=None, parent=None) -> None:
        super().__init__("settingsPage", "设置")
        self.services = services
        self.on_changed = on_changed
        self.forms: dict[str, SchemaForm] = {}
        cfg = services.cfg
        top = Card()
        reset = PushButton(FIF.SYNC, "恢复默认")
        reset.clicked.connect(self.reset)
        open_cfg = PushButton(FIF.DOCUMENT, "打开 config.yaml 所在目录")
        open_cfg.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(services.config.path.parent))))
        top.lay.addLayout(row(reset, open_cfg, None))
        if services.config.problems:
            p = CaptionLabel("读取配置时的问题：\n" + "\n".join(services.config.problems))
            p.setWordWrap(True)
            top.lay.addWidget(p)
        self.body.addWidget(top)

        self._section("app", "通用", APP_FIELDS)
        self._qwen_section()
        self._glossary_section()
        self._section("segmentation", "音频与分段", SEG_FIELDS)
        self._diarization_section()
        vp = self._section("voiceprint", "声纹识别", VP_FIELDS)
        b1 = PushButton(FIF.PEOPLE, "打开声纹库")
        b1.clicked.connect(lambda: open_voiceprints and open_voiceprints())
        b2 = PushButton(FIF.SPEED_HIGH, "用库内样本校准阈值")
        b2.clicked.connect(self.calibrate)
        vp.lay.addLayout(row(b1, b2, None))
        self._llm_section()
        self._section("text", "文本处理（规则化，未配置大模型时使用）", TEXT_FIELDS)
        self._section("output", "导出", OUT_FIELDS)
        self._realtime_section()
        api = Card("API 模型")
        b = PushButton(FIF.CLOUD, "管理 API 模型")
        b.clicked.connect(lambda: open_api_models and open_api_models())
        api.lay.addLayout(row(b, None))
        self.body.addWidget(api)
        self._section("batch", "批量处理默认选项", BATCH_FIELDS)
        self._section("logging", "日志", LOG_FIELDS)
        self._diagnostics_section()
        self.load(cfg)

    # ------------------------------------------------------------------ generic sections
    def _section(self, key: str, title: str, fields, description: str = "") -> Card:
        card = Card(title, description)
        form = SchemaForm(fields)
        self.forms[key] = form
        card.lay.addWidget(form)
        save = PrimaryPushButton(FIF.SAVE, "保存")
        save.clicked.connect(lambda _c=False, k=key: self.save_section(k))
        card.lay.addLayout(row(None, save))
        self.body.addWidget(card)
        return card

    def load(self, cfg: AppConfig) -> None:
        for key, form in self.forms.items():
            form.set_values(getattr(cfg, key).model_dump(mode="json"))
        self._load_special(cfg)

    def save_section(self, key: str) -> None:
        try:
            vals = self.forms[key].values()
        except ValueError as e:
            warning(self, "设置有误", str(e))
            return

        def mut(c: AppConfig) -> None:
            sec_obj = getattr(c, key)
            data = sec_obj.model_dump()
            _deep(data, vals)
            setattr(c, key, type(sec_obj).model_validate(data))

        try:
            cfg = self.services.config.update(mut)
        except Exception as e:  # noqa: BLE001
            error(self, "保存失败", str(e))
            return
        if key == "app":
            apply_theme(cfg.app.theme)
        if key == "logging":
            from asr_dashboard.app import setup_logging

            setup_logging(cfg)
        success(self, "已保存")
        if self.on_changed:
            self.on_changed()

    def reset(self) -> None:
        if MessageBox("恢复默认", "所有设置恢复为默认值（API 模型和词库保留）。确定？", self.window()).exec():
            cfg = self.services.config.reset_defaults(keep_api_models=True)
            self.load(cfg)
            apply_theme(cfg.app.theme)
            if self.on_changed:
                self.on_changed()
            success(self, "已恢复默认设置")

    # ------------------------------------------------------------------ qwen3
    def _qwen_section(self) -> None:
        card = self._section("qwen3", "本地模型（Qwen3-ASR）", Qwen3LocalEngine.settings_schema(),
                             "本地只提供 Qwen3-ASR-1.7B / 0.6B 两个识别模型，以及 Qwen3-ForcedAligner-0.6B 对齐器。")
        self.dl_bar = ProgressBar()
        self.dl_bar.setRange(0, 1000)
        self.dl_label = CaptionLabel("")
        dl = PushButton(FIF.DOWNLOAD, "下载模型")
        dl.clicked.connect(self.download_models)
        warm = PushButton(FIF.SPEED_HIGH, "预热（加载到内存）")
        warm.clicked.connect(self.prewarm)
        unload = PushButton(FIF.BROOM, "卸载并清理显存")
        unload.clicked.connect(lambda: (self.services.unload_all(), success(self, "已卸载")))
        scan = PushButton(FIF.SEARCH, "查看缓存")
        scan.clicked.connect(self.scan_cache)
        card.lay.addWidget(StrongBodyLabel("模型管理"))
        card.lay.addLayout(row(dl, warm, unload, scan, None))
        card.lay.addWidget(self.dl_bar)
        card.lay.addWidget(self.dl_label)
        self.cache_list = ListWidget()
        self.cache_list.setFixedHeight(110)
        del_btn = PushButton(FIF.DELETE, "删除选中缓存")
        del_btn.clicked.connect(self.delete_cache)
        card.lay.addWidget(self.cache_list)
        card.lay.addLayout(row(del_btn, None))

    def download_models(self) -> None:
        from asr_dashboard.engines.qwen3 import model_manager

        q = self.services.cfg.qwen3
        if q.source == "local":
            info(self, "模型来源为本地目录，无需下载")
            return
        self.dl_label.setText("开始下载…")
        run_bg(lambda progress: model_manager.download(q.variant, q.backend, q.source, q.aligner.enabled,
                                                       q.cache_dir or None, progress),
               on_ok=lambda paths: (self.dl_label.setText("下载完成：" + "；".join(paths)), self.scan_cache()),
               on_err=lambda m, tb: (self.dl_label.setText(f"下载失败：{m}"), error(self, "下载失败", m)),
               on_progress=lambda f, m: (self.dl_bar.setValue(int(f * 1000)), self.dl_label.setText(m)), parent=self)

    def prewarm(self) -> None:
        eng = self.services.engine(f"qwen3-local:{self.services.cfg.qwen3.variant}")
        self.dl_label.setText("正在加载模型…（日志中会打印最终的 from_pretrained 参数）")
        run_bg(eng.load, on_ok=lambda _r: (self.dl_label.setText(f"模型已加载到 {eng.effective_device}"),
                                           [warning(self, "提示", m) for m in eng.info_messages]),
               on_err=lambda m, tb: (self.dl_label.setText(f"加载失败：{m}"), error(self, "加载失败", m)), parent=self)

    def scan_cache(self) -> None:
        from asr_dashboard.engines.qwen3 import model_manager

        def done(items):
            self._cache = items
            self.cache_list.clear()
            for c in items:
                self.cache_list.addItem(QListWidgetItem(f"{c.repo_id} · {c.size_gb:.2f} GB · {c.source} · {c.path}"))
            if not items:
                self.cache_list.addItem("（没有找到已下载的模型）")

        run_bg(model_manager.list_cached, self.services.cfg.qwen3.cache_dir or None, on_ok=done,
               on_err=lambda m, tb: error(self, "扫描失败", m), parent=self)

    def delete_cache(self) -> None:
        from asr_dashboard.engines.qwen3 import model_manager

        r = self.cache_list.currentRow()
        items = getattr(self, "_cache", [])
        if 0 <= r < len(items) and MessageBox("删除缓存", f"删除 {items[r].repo_id}？", self.window()).exec():
            self.services.unload_all()
            model_manager.delete_cached(items[r])
            self.scan_cache()

    # ------------------------------------------------------------------ glossaries
    def _glossary_section(self) -> None:
        card = Card("上下文与词库", "词库是一组词条（专有名词、人名、术语）加可选说明；首页可多选启用。默认上下文在“本地模型”中设置。")
        self.gl_list = ListWidget()
        self.gl_list.setFixedHeight(110)
        self.gl_list.currentRowChanged.connect(self._show_glossary)
        self.gl_name = LineEdit()
        self.gl_desc = LineEdit()
        self.gl_desc.setPlaceholderText("说明（可选，作为背景信息）")
        self.gl_terms = PlainTextEdit()
        self.gl_terms.setPlaceholderText("每行一个词条")
        self.gl_terms.setFixedHeight(110)
        add = PushButton(FIF.ADD, "新建词库")
        add.clicked.connect(self._add_glossary)
        save = PrimaryPushButton(FIF.SAVE, "保存词库")
        save.clicked.connect(self._save_glossary)
        rm = PushButton(FIF.DELETE, "删除词库")
        rm.clicked.connect(self._del_glossary)
        card.lay.addWidget(self.gl_list)
        card.lay.addLayout(row(BodyLabel("名称"), self.gl_name, self.gl_desc))
        card.lay.addWidget(self.gl_terms)
        card.lay.addLayout(row(add, save, rm, None))
        self.body.addWidget(card)

    def _refresh_glossaries(self) -> None:
        self.gl_list.blockSignals(True)
        self.gl_list.clear()
        for g in self.services.cfg.glossaries:
            self.gl_list.addItem(f"{g.name}（{len(g.terms)} 条）")
        self.gl_list.blockSignals(False)
        if self.services.cfg.glossaries:
            self.gl_list.setCurrentRow(0)

    def _show_glossary(self, r: int) -> None:
        gls = self.services.cfg.glossaries
        if 0 <= r < len(gls):
            g = gls[r]
            self.gl_name.setText(g.name)
            self.gl_desc.setText(g.description)
            self.gl_terms.setPlainText("\n".join(g.terms))

    def _add_glossary(self) -> None:
        self.services.config.update(lambda c: c.glossaries.append(Glossary(name="新词库")))
        self._refresh_glossaries()
        self.gl_list.setCurrentRow(len(self.services.cfg.glossaries) - 1)
        self.on_changed and self.on_changed()

    def _save_glossary(self) -> None:
        r = self.gl_list.currentRow()
        if r < 0:
            return
        terms = [t.strip() for t in self.gl_terms.toPlainText().splitlines() if t.strip()]

        def mut(c):
            g = c.glossaries[r]
            c.glossaries[r] = Glossary(id=g.id, name=self.gl_name.text().strip() or "词库",
                                       description=self.gl_desc.text(), terms=terms)

        self.services.config.update(mut)
        self._refresh_glossaries()
        self.gl_list.setCurrentRow(r)
        success(self, "词库已保存")
        self.on_changed and self.on_changed()

    def _del_glossary(self) -> None:
        r = self.gl_list.currentRow()
        if r >= 0:
            self.services.config.update(lambda c: c.glossaries.pop(r))
            self._refresh_glossaries()
            self.on_changed and self.on_changed()

    # ------------------------------------------------------------------ diarization
    def _diarization_section(self) -> None:
        card = self._section("diarization", "说话人分离", DIAR_FIELDS)
        card.lay.addWidget(StrongBodyLabel("pyannote Community-1 首次使用引导"))
        card.lay.addWidget(CaptionLabel("1. 填写 HuggingFace Token  2. 打开模型页面登录并同意使用条款（受限模型）"
                                        "  3. 点击下载到本地  4. 之后完全离线运行，不再需要 Token"))
        self.hf_token = PasswordLineEdit()
        self.hf_token.setPlaceholderText("hf_…")
        link = HyperlinkButton(pyannote_backend.MODEL_PAGE, "打开模型页面")
        link2 = HyperlinkButton(pyannote_backend.TOKEN_PAGE, "创建 Token")
        self.pa_dir = LineEdit()
        self.pa_dir.setPlaceholderText(str(pyannote_backend.default_model_dir()))
        dl = PrimaryPushButton(FIF.DOWNLOAD, "下载 Community-1")
        dl.clicked.connect(self.download_pyannote)
        self.pa_status = CaptionLabel("")
        card.lay.addLayout(row(BodyLabel("HF Token"), self.hf_token, link2, link))
        card.lay.addLayout(row(BodyLabel("本地目录"), self.pa_dir, dl))
        card.lay.addWidget(self.pa_status)
        card.lay.addWidget(StrongBodyLabel("DiariZen（独立隔离环境）"))
        lic = CaptionLabel(diarizen_backend.LICENSE_TEXT)
        lic.setWordWrap(True)
        card.lay.addWidget(lic)
        self.dz_license = CheckBox("我已阅读并确认 DiariZen 模型权重仅限非商业用途")
        self.dz_license.toggled.connect(self._accept_license)
        self.dz_cuda = CheckBox("安装 CUDA 版 PyTorch（取消则安装 CPU 版）")
        self.dz_cuda.setChecked(True)
        inst = PushButton(FIF.DOWNLOAD, "安装 DiariZen 环境")
        inst.clicked.connect(self.install_diarizen)
        uninst = PushButton(FIF.DELETE, "卸载")
        uninst.clicked.connect(self.uninstall_diarizen)
        self.dz_status = CaptionLabel("")
        card.lay.addWidget(self.dz_license)
        card.lay.addLayout(row(self.dz_cuda, inst, uninst, None))
        self.dz_log = PlainTextEdit()
        self.dz_log.setReadOnly(True)
        self.dz_log.setFixedHeight(120)
        card.lay.addWidget(self.dz_status)
        card.lay.addWidget(self.dz_log)

    def download_pyannote(self) -> None:
        tok = self.hf_token.text().strip()
        if tok:
            self.services.secrets.set(sec.HF_TOKEN, tok)
        tok = tok or self.services.secrets.get(sec.HF_TOKEN)
        if not tok:
            warning(self, "请先填写 HuggingFace Token")
            return
        target = self.pa_dir.text().strip() or None
        self.pa_status.setText("下载中…")

        def ok(path):
            self.services.config.update(lambda c: setattr(c.diarization.pyannote, "model_dir", str(path)))
            self.pa_status.setText(f"已下载到 {path}，之后离线运行")
            success(self, "pyannote 模型已就绪")

        run_bg(pyannote_backend.download, tok, target, on_ok=ok,
               on_err=lambda m, tb: (self.pa_status.setText(m), error(self, "下载失败", m)), parent=self)

    def _accept_license(self, checked: bool) -> None:
        self.services.config.update(lambda c: setattr(c.diarization.diarizen, "license_accepted", bool(checked)))

    def install_diarizen(self) -> None:
        cfg = self.services.cfg
        inst = diarizen_backend.DiarizenInstaller(cfg.diarization.diarizen.env_dir or None, self.dz_cuda.isChecked())
        self.dz_log.clear()
        from PyQt6.QtCore import QObject, pyqtSignal

        class Bridge(QObject):
            line = pyqtSignal(str)

        self._bridge = Bridge()
        self._bridge.line.connect(self.dz_log.appendPlainText)
        inst.log = self._bridge.line.emit
        self.dz_status.setText("安装中（需要 uv 和 git，可能需要较长时间）…")
        run_bg(lambda progress: inst.install(progress=progress),
               on_ok=lambda _r: (self.dz_status.setText("DiariZen 环境安装完成"), success(self, "DiariZen 已安装")),
               on_err=lambda m, tb: (self.dz_status.setText(f"安装失败：{m}"), error(self, "安装失败", m)),
               on_progress=lambda f, m: self.dz_status.setText(m), parent=self)

    def uninstall_diarizen(self) -> None:
        if MessageBox("卸载 DiariZen", "删除 DiariZen 隔离环境？", self.window()).exec():
            diarizen_backend.DiarizenInstaller(self.services.cfg.diarization.diarizen.env_dir or None).uninstall()
            self.dz_status.setText("已卸载")

    def calibrate(self) -> None:
        from asr_dashboard.core.voiceprints import matcher

        emb = self.services.embedder(self.services.cfg)
        per = self.services.voiceprints.sample_embeddings(emb.model_id)
        cal = matcher.calibrate(per)
        if cal.threshold is None:
            warning(self, "无法校准", cal.message)
            return
        if MessageBox("阈值校准", cal.message + "\n\n是否采用推荐值？", self.window()).exec():
            self.services.config.update(lambda c: setattr(c.voiceprint, "match_threshold", round(cal.threshold, 2)))
            self.load(self.services.cfg)

    # ------------------------------------------------------------------ LLM
    def _llm_section(self) -> None:
        card = Card("智能后处理（大模型）", "配置一个 OpenAI 兼容的大模型 API。保存时会验证；验证通过后，文字整理交给大模型，否则使用规则化处理。")
        self.llm_url = LineEdit()
        self.llm_url.setPlaceholderText("https://api.deepseek.com/v1 · https://dashscope.aliyuncs.com/compatible-mode/v1 · http://localhost:11434/v1")
        self.llm_key = PasswordLineEdit()
        self.llm_key.setPlaceholderText("API Key（localhost 可留空）")
        self.llm_model = LineEdit()
        self.llm_model.setPlaceholderText("deepseek-chat / qwen-plus / qwen3:8b")
        self.llm_enabled = CheckBox("启用")
        self.llm_cleanup = CheckBox("文字整理")
        self.llm_summary = CheckBox("生成摘要和要点")
        self.llm_guess = CheckBox("推测说话人身份")
        card.lay.addLayout(row(BodyLabel("API 地址"), self.llm_url))
        card.lay.addLayout(row(BodyLabel("API Key"), self.llm_key, BodyLabel("模型名"), self.llm_model))
        card.lay.addLayout(row(self.llm_enabled, BodyLabel("处理内容："), self.llm_cleanup, self.llm_summary, self.llm_guess, None))
        self.llm_adv = SchemaForm(LLM_ADV)
        card.lay.addWidget(CaptionLabel("高级"))
        card.lay.addWidget(self.llm_adv)
        save = PrimaryPushButton(FIF.ACCEPT, "保存并验证")
        save.clicked.connect(self.save_llm)
        clear = PushButton(FIF.DELETE, "清除配置")
        clear.clicked.connect(self.clear_llm)
        self.llm_status = StrongBodyLabel("")
        card.lay.addLayout(row(save, clear, self.llm_status, None))
        self.body.addWidget(card)

    def save_llm(self) -> None:
        from asr_dashboard.text.llm_client import is_local, validate

        url, key, model = self.llm_url.text().strip(), self.llm_key.text().strip(), self.llm_model.text().strip()
        try:
            adv = self.llm_adv.values()
        except ValueError as e:
            warning(self, "设置有误", str(e))
            return
        self.llm_status.setText("验证中…")

        def ok(res):
            if not res.ok:
                self.llm_status.setText(f"不可用：{res.message[:120]}")
                error(self, "验证失败，未保存", f"{res.message[:200]}\n{res.hint}", 10000)
                return
            first_time = not self.services.cfg.llm.privacy_acknowledged and not is_local(url)
            if first_time and not MessageBox("隐私提示", f"启用后，转录文字会发送到 {url}。确定继续？",
                                             self.window()).exec():
                self.llm_status.setText("已取消")
                return

            def mut(c: AppConfig):
                data = c.llm.model_dump()
                data.update(adv)
                data.update(base_url=url, model=model, validated=True, json_mode=res.json_mode,
                            enabled=self.llm_enabled.isChecked(),
                            tasks={"cleanup": self.llm_cleanup.isChecked(), "summary": self.llm_summary.isChecked(),
                                   "speaker_guess": self.llm_guess.isChecked()},
                            privacy_acknowledged=c.llm.privacy_acknowledged or first_time or is_local(url))
                c.llm = type(c.llm).model_validate(data)

            self.services.secrets.set(sec.LLM_KEY, key)
            self.services.config.update(mut)
            self.llm_status.setText(f"可用 · {model} · 响应 {res.latency:.2f}s" + ("" if res.json_mode else " · 不支持 JSON 模式，改用提示词约束"))
            success(self, "大模型已启用", model)
            self.on_changed and self.on_changed()

        run_bg(validate, url, key, model, on_ok=ok, on_err=lambda m, tb: self.llm_status.setText(f"不可用：{m}"), parent=self)

    def clear_llm(self) -> None:
        self.services.secrets.delete(sec.LLM_KEY)
        self.services.config.update(lambda c: setattr(c, "llm", type(c.llm)()))
        self.load(self.services.cfg)
        self.on_changed and self.on_changed()

    # ------------------------------------------------------------------ realtime
    def _realtime_section(self) -> None:
        card = Card("实时转录服务", "连接外部流式识别服务（自部署 vLLM 或阿里云百炼）。验证通过后才会开启“实时转录”。")
        from qfluentwidgets import ComboBox

        self.rt_proto = ComboBox()
        self.rt_proto.addItem("vLLM /v1/realtime（自部署）", userData="vllm_realtime")
        self.rt_proto.addItem("阿里云百炼实时语音识别", userData="dashscope_realtime")
        self.rt_url = LineEdit()
        self.rt_url.setPlaceholderText("ws://192.168.1.10:8000/v1/realtime 或 wss://dashscope.aliyuncs.com/api-ws/v1/realtime")
        self.rt_key = PasswordLineEdit()
        self.rt_key.setPlaceholderText("API Key（可选）")
        self.rt_model = LineEdit()
        self.rt_model.setPlaceholderText("Qwen/Qwen3-ASR-1.7B 或 qwen3-asr-flash-realtime")
        self.rt_lang = ComboBox()
        self.rt_lang.addItem("自动", userData="auto")
        for c, n in QWEN_LANGUAGES.items():
            self.rt_lang.addItem(f"{n} ({c})", userData=c)
        self.rt_enabled = CheckBox("启用")
        card.lay.addLayout(row(BodyLabel("协议"), self.rt_proto, self.rt_enabled, None))
        card.lay.addLayout(row(BodyLabel("服务地址"), self.rt_url))
        card.lay.addLayout(row(BodyLabel("API Key"), self.rt_key, BodyLabel("模型名"), self.rt_model, BodyLabel("语言"), self.rt_lang))
        self.rt_adv = SchemaForm(RT_ADV)
        card.lay.addWidget(CaptionLabel("高级"))
        card.lay.addWidget(self.rt_adv)
        save = PrimaryPushButton(FIF.ACCEPT, "保存并验证")
        save.clicked.connect(self.save_realtime)
        clear = PushButton(FIF.DELETE, "清除配置")
        clear.clicked.connect(self.clear_realtime)
        doc = HyperlinkButton("https://github.com/PXzy98/Asr-dashboard#部署实时转录服务", "服务端部署说明")
        self.rt_status = StrongBodyLabel("")
        card.lay.addLayout(row(save, clear, doc, self.rt_status, None))
        self.body.addWidget(card)

    def save_realtime(self) -> None:
        from asr_dashboard.realtime.validate import validate

        proto, url, key = self.rt_proto.currentData(), self.rt_url.text().strip(), self.rt_key.text().strip()
        model, lang = self.rt_model.text().strip(), self.rt_lang.currentData()
        try:
            adv = self.rt_adv.values()
        except ValueError as e:
            warning(self, "设置有误", str(e))
            return
        self.rt_status.setText("验证中（发送内置测试语音）…")

        def ok(res):
            if not res.ok:
                self.rt_status.setText(f"不可用：{res.message[:120]}")
                error(self, "验证失败，未保存", f"{res.message[:200]}\n{res.hint}", 10000)
                return

            def mut(c: AppConfig):
                data = c.realtime.model_dump()
                data.update(adv)
                data.update(protocol=proto, url=url, model=model, language=lang, validated=True,
                            enabled=self.rt_enabled.isChecked())
                c.realtime = type(c.realtime).model_validate(data)

            self.services.secrets.set(sec.REALTIME_KEY, key)
            self.services.config.update(mut)
            lat = f"首字 {res.first_text_latency:.2f}s · " if res.first_text_latency is not None else ""
            self.rt_status.setText(f"可用 · {lat}总耗时 {res.total:.2f}s · 识别：{res.text[:30]}")
            success(self, "实时转录已启用")
            self.on_changed and self.on_changed()

        run_bg(validate, proto, url, key, model, None if lang == "auto" else lang, adv.get("chunk_ms", 100),
               on_ok=ok, on_err=lambda m, tb: self.rt_status.setText(f"不可用：{m}"), parent=self)

    def clear_realtime(self) -> None:
        self.services.secrets.delete(sec.REALTIME_KEY)
        self.services.config.update(lambda c: setattr(c, "realtime", type(c.realtime)()))
        self.load(self.services.cfg)
        self.on_changed and self.on_changed()

    # ------------------------------------------------------------------ diagnostics
    def _diagnostics_section(self) -> None:
        card = Card("环境诊断")
        self.diag = PlainTextEdit()
        self.diag.setReadOnly(True)
        self.diag.setFixedHeight(200)
        run_btn = PushButton(FIF.SYNC, "重新检测")
        run_btn.clicked.connect(self.run_diagnostics)
        copy = PushButton(FIF.COPY, "复制")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.diag.toPlainText()))
        card.lay.addLayout(row(run_btn, copy, None))
        card.lay.addWidget(self.diag)
        self.body.addWidget(card)

    def run_diagnostics(self) -> None:
        from asr_dashboard.core import diagnostics

        self.diag.setPlainText("检测中…")
        run_bg(diagnostics.run, self.services.cfg, self.services.secrets, on_ok=self.diag.setPlainText,
               on_err=lambda m, tb: self.diag.setPlainText(tb), parent=self)

    # ------------------------------------------------------------------ load special widgets
    def _load_special(self, cfg: AppConfig) -> None:
        self._refresh_glossaries()
        self.hf_token.setText(self.services.secrets.get(sec.HF_TOKEN))
        self.pa_dir.setText(cfg.diarization.pyannote.model_dir)
        ok, reason = pyannote_backend.availability(cfg.diarization.pyannote.model_dir or None)
        self.pa_status.setText("pyannote：可用（离线）" if ok else f"pyannote：{reason}")
        self.dz_license.blockSignals(True)
        self.dz_license.setChecked(cfg.diarization.diarizen.license_accepted)
        self.dz_license.blockSignals(False)
        ok, reason = diarizen_backend.availability(cfg.diarization.diarizen.env_dir or None, True)
        self.dz_status.setText("DiariZen 环境：已安装" if ok else f"DiariZen：{reason}")
        llm = cfg.llm
        self.llm_url.setText(llm.base_url)
        self.llm_key.setText(self.services.secrets.get(sec.LLM_KEY))
        self.llm_model.setText(llm.model)
        self.llm_enabled.setChecked(llm.enabled)
        self.llm_cleanup.setChecked(llm.tasks.cleanup)
        self.llm_summary.setChecked(llm.tasks.summary)
        self.llm_guess.setChecked(llm.tasks.speaker_guess)
        self.llm_adv.set_values(llm.model_dump(mode="json"))
        self.llm_status.setText("未配置" if not llm.base_url else
                                (f"可用 · {llm.model}" if llm.validated else "未验证") + ("" if llm.enabled else "（已关闭，使用规则）"))
        rt = cfg.realtime
        self.rt_proto.setCurrentIndex(0 if rt.protocol == "vllm_realtime" else 1)
        self.rt_url.setText(rt.url)
        self.rt_key.setText(self.services.secrets.get(sec.REALTIME_KEY))
        self.rt_model.setText(rt.model)
        idx = next((i for i in range(self.rt_lang.count()) if self.rt_lang.itemData(i) == rt.language), 0)
        self.rt_lang.setCurrentIndex(idx)
        self.rt_enabled.setChecked(rt.enabled)
        self.rt_adv.set_values(rt.model_dump(mode="json"))
        self.rt_status.setText("未配置" if not rt.url else ("可用" if rt.validated else "未验证")
                               + ("" if rt.enabled else "（已关闭）"))


def _deep(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict) and k != "filler_words":
            _deep(dst[k], v)
        else:
            dst[k] = v


def apply_theme(theme: str) -> None:
    setTheme({"light": Theme.LIGHT, "dark": Theme.DARK}.get(theme, Theme.AUTO))

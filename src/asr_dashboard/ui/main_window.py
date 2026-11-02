"""Main window: navigation and page assembly only."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import FluentWindow, MessageBox, NavigationItemPosition

from asr_dashboard.core import batch as batchmod
from asr_dashboard.core.pipeline import TaskRequest
from asr_dashboard.core.tasks import TranscriptionWorker
from asr_dashboard.ui.pages.api_models_page import ApiModelsPage
from asr_dashboard.ui.pages.batch_page import BatchPage
from asr_dashboard.ui.pages.history_page import HistoryPage
from asr_dashboard.ui.pages.home_page import HomePage
from asr_dashboard.ui.pages.realtime_page import RealtimePage
from asr_dashboard.ui.pages.results_page import ResultsPage
from asr_dashboard.ui.pages.settings_page import SettingsPage
from asr_dashboard.ui.pages.voiceprints_page import VoiceprintsPage
from asr_dashboard.ui.widgets.common import error, info, success, warning
from asr_dashboard.ui.widgets.error_dialog import show_error


class MainWindow(FluentWindow):
    def __init__(self, services) -> None:
        super().__init__()
        self.services = services
        self.setWindowTitle("ASR Dashboard")
        self.resize(1280, 860)
        self.setMinimumSize(QSize(980, 640))

        self.home = HomePage(services, self)
        self.batch = BatchPage(services, self.home.options, self)
        self.realtime = RealtimePage(services, self)
        self.results = ResultsPage(services, self)
        self.history = HistoryPage(services, self)
        self.voiceprints = VoiceprintsPage(services, self)
        self.api_models = ApiModelsPage(services, on_changed=self.refresh_all, parent=self)
        self.settings = SettingsPage(services, on_changed=self.refresh_all,
                                     open_voiceprints=lambda: self.switchTo(self.voiceprints),
                                     open_api_models=lambda: self.switchTo(self.api_models), parent=self)

        self.addSubInterface(self.home, FIF.HOME, "单文件")
        self.addSubInterface(self.batch, FIF.FOLDER, "批量处理")
        self.addSubInterface(self.realtime, FIF.MICROPHONE, "实时转录")
        self.addSubInterface(self.results, FIF.DOCUMENT, "结果")
        self.addSubInterface(self.history, FIF.HISTORY, "历史")
        self.addSubInterface(self.voiceprints, FIF.PEOPLE, "声纹库")
        self.addSubInterface(self.api_models, FIF.CLOUD, "API 模型")
        self.addSubInterface(self.settings, FIF.SETTING, "设置", NavigationItemPosition.BOTTOM)

        self.home.result_ready.connect(self._show_output)
        self.history.open_result.connect(lambda r, hid: (self.results.show_result(r, hid), self.switchTo(self.results)))
        self.realtime.refine_requested.connect(self._refine)
        self.stackedWidget.currentChanged.connect(self._page_changed)
        self.update_realtime_nav()
        if services.config.problems:
            warning(self, "配置文件有问题，已使用默认值", "；".join(services.config.problems)[:300], 10000)

    # ------------------------------------------------------------------ navigation
    def update_realtime_nav(self) -> None:
        item = self.navigationInterface.widget(self.realtime.objectName())
        active = self.services.cfg.realtime_active
        if item is not None:
            item.setEnabled(active)
            item.setToolTip("" if active else "请先在 设置 → 实时转录服务 中配置")

    def refresh_all(self) -> None:
        self.home.refresh()
        self.batch.refresh()
        self.realtime.refresh()
        self.api_models.refresh() if hasattr(self, "api_models") else None
        self.update_realtime_nav()

    def _page_changed(self, _i: int) -> None:
        w = self.stackedWidget.currentWidget()
        if w is self.history:
            self.history.refresh()
        elif w is self.voiceprints:
            self.voiceprints.refresh()

    def _show_output(self, out) -> None:
        self.results.show_result(out.result)
        self.switchTo(self.results)

    # ------------------------------------------------------------------ realtime refine
    def _refine(self, wav: str, draft: str) -> None:
        cfg = self.home.options.build_config()
        req = TaskRequest(wav, "", self.home.options.selected_glossaries(), language=self.home.options.language_code(),
                          output_dir=str(Path(wav).parent), output_stem=Path(wav).stem + "_refined")
        w = TranscriptionWorker(self.services, self.home.options.engine_id(), cfg, req, self)
        info(self, "正在精修", "用当前首页引擎完整处理刚才的录音…", 5000)

        def done(out):
            out.result.draft_text = draft
            self.results.show_result(out.result)
            self.switchTo(self.results)
            success(self, "精修完成", "实时草稿已保留在 JSON 的 draft_text 字段中")

        w.succeeded.connect(done)
        w.failed.connect(lambda m, tb: show_error(self, "精修失败", m, tb))
        self._refine_worker = w
        w.start()

    # ------------------------------------------------------------------ startup / shutdown
    def check_unfinished_batches(self) -> None:
        for m in batchmod.find_unfinished():
            n = m.next_index()
            box = MessageBox("发现未完成的批次",
                             f"文件夹：{m.folder}\n已完成 {sum(i.status == batchmod.DONE for i in m.items)} / {len(m.items)}。\n"
                             f"是否从第 {n + 1} 个文件继续？（选择“取消”将放弃该批次）", self)
            box.yesButton.setText("继续")
            box.cancelButton.setText("放弃")
            if box.exec():
                self.switchTo(self.batch)
                self.batch.resume(m)
                return
            batchmod.abandon(m)

    def closeEvent(self, e) -> None:  # noqa: N802
        busy = self.home.is_busy() or self.batch.is_busy() or self.realtime.is_busy()
        if busy and not MessageBox("确认退出", "仍有任务在运行，确定退出？（批量进度已保存，可下次继续）", self).exec():
            e.ignore()
            return
        try:
            if self.home.worker:
                self.home.worker.cancel()
            if self.batch.worker:
                self.batch.worker.cancel()
            if self.realtime.is_busy():
                self.realtime.stop()
            self.results.stop_playback()
            for w in (self.home.worker, self.batch.worker):
                if w is not None:
                    w.wait(3000)
            self.services.unload_all()
        except Exception as ex:  # noqa: BLE001
            error(self, "退出时出错", str(ex))
        super().closeEvent(e)

"""Application start-up: logging, global exception hook, theme, self check, main window."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import traceback
from pathlib import Path

from asr_dashboard import paths
from asr_dashboard.redact import RedactingFilter, redact

log = logging.getLogger("asr_dashboard")


def setup_logging(cfg=None) -> Path:
    lc = cfg.logging if cfg is not None else None
    log_dir = Path(lc.log_dir) if lc and lc.log_dir else paths.app_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_asr_dashboard", False):
            root.removeHandler(h)
    level = getattr(logging, lc.level if lc else "INFO")
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(log_dir / "asr-dashboard.log", encoding="utf-8",
                                              maxBytes=(lc.max_file_size_mb if lc else 10) * 1024 * 1024,
                                              backupCount=lc.backup_count if lc else 5)
    sh = logging.StreamHandler(sys.stderr)
    for h in (fh, sh):
        h.setFormatter(fmt)
        h.addFilter(RedactingFilter())
        h._asr_dashboard = True  # type: ignore[attr-defined]
        root.addHandler(h)
    for noisy in ("httpx", "httpcore", "urllib3", "websockets", "numba", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_dir


def install_excepthook(get_window) -> None:
    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = redact("".join(traceback.format_exception(exc_type, exc, tb)))
        log.error("未处理的异常：\n%s", text)
        try:
            from asr_dashboard.ui.widgets.error_dialog import show_error

            show_error(get_window(), "程序出错", redact(str(exc)) or exc_type.__name__, text)
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook
    threading.excepthook = lambda args: log.error("线程异常：%s", redact("".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))))


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    from PyQt6.QtWidgets import QApplication

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("ASR Dashboard")

    from asr_dashboard.config.manager import ConfigManager

    setup_logging(None)
    config = ConfigManager()
    setup_logging(config.config)
    from asr_dashboard.i18n import set_language

    set_language(config.config.app.ui_language)
    log.info("ASR Dashboard 启动，配置文件：%s", config.path)

    from asr_dashboard.services import Services
    from asr_dashboard.ui.main_window import MainWindow
    from asr_dashboard.ui.pages.settings_page import apply_theme

    apply_theme(config.config.app.theme)
    services = Services(config)
    window = MainWindow(services)
    install_excepthook(lambda: window)
    window.show()

    from PyQt6.QtCore import QTimer

    from asr_dashboard.core import audio

    def after_show() -> None:
        if not audio.ffmpeg_path():
            from asr_dashboard.ui.widgets.error_dialog import show_error

            show_error(window, "未检测到 FFmpeg", audio.FFMPEG_HELP)
        window.settings.run_diagnostics()
        window.check_unfinished_batches()

    QTimer.singleShot(300, after_show)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

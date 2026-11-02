"""Small shared UI helpers: background function runner, InfoBar shortcuts, scrollable page base."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    InfoBar,
    InfoBarPosition,
    ScrollArea,
    SimpleCardWidget,
    StrongBodyLabel,
    TitleLabel,
)

from asr_dashboard.redact import redact

log = logging.getLogger(__name__)


class FunctionWorker(QThread):
    """Run ``fn(progress=callable)`` in a thread; progress(frac, msg) is forwarded as a signal."""

    progress = pyqtSignal(float, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, fn: Callable, *args, with_progress: bool = False, parent: QObject | None = None, **kwargs):
        super().__init__(parent)
        self.fn, self.args, self.kwargs, self.with_progress = fn, args, kwargs, with_progress

    def run(self) -> None:
        try:
            if self.with_progress:
                self.kwargs["progress"] = lambda f, m="": self.progress.emit(float(f), str(m))
            self.succeeded.emit(self.fn(*self.args, **self.kwargs))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(redact(str(e)), redact(traceback.format_exc()))


_workers: set[QThread] = set()


def run_bg(fn: Callable, *args, on_ok: Callable | None = None, on_err: Callable | None = None,
           on_progress: Callable | None = None, parent=None, **kwargs) -> FunctionWorker:
    """Fire-and-forget background call that keeps a reference until it finishes."""
    w = FunctionWorker(fn, *args, with_progress=on_progress is not None, parent=parent, **kwargs)
    if on_ok:
        w.succeeded.connect(on_ok)
    if on_err:
        w.failed.connect(on_err)
    if on_progress:
        w.progress.connect(on_progress)
    _workers.add(w)
    w.finished.connect(lambda: _workers.discard(w))
    w.start()
    return w


def window_of(widget: QWidget) -> QWidget:
    return widget.window() if widget is not None else None


def info(widget, title: str, text: str = "", duration: int = 3000) -> None:
    InfoBar.info(title, text, parent=window_of(widget), position=InfoBarPosition.TOP_RIGHT, duration=duration)


def success(widget, title: str, text: str = "", duration: int = 3000) -> None:
    InfoBar.success(title, text, parent=window_of(widget), position=InfoBarPosition.TOP_RIGHT, duration=duration)


def warning(widget, title: str, text: str = "", duration: int = 6000) -> None:
    InfoBar.warning(title, text, parent=window_of(widget), position=InfoBarPosition.TOP_RIGHT, duration=duration)


def error(widget, title: str, text: str = "", duration: int = -1) -> None:
    InfoBar.error(title, redact(text), parent=window_of(widget), position=InfoBarPosition.TOP_RIGHT,
                  duration=duration)


class Page(ScrollArea):
    """Scrollable page with a title; add content to ``self.body`` (QVBoxLayout)."""

    def __init__(self, object_name: str, title: str, subtitle: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.view = QWidget()
        self.view.setObjectName("view")
        self.body = QVBoxLayout(self.view)
        self.body.setContentsMargins(28, 20, 28, 28)
        self.body.setSpacing(14)
        self.title = TitleLabel(title)
        self.body.addWidget(self.title)
        if subtitle:
            cap = CaptionLabel(subtitle)
            cap.setWordWrap(True)
            self.body.addWidget(cap)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.enableTransparentBackground()


class Card(SimpleCardWidget):
    """Titled card with a vertical layout ``self.lay``."""

    def __init__(self, title: str = "", description: str = "", parent=None) -> None:
        super().__init__(parent)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(18, 14, 18, 16)
        self.lay.setSpacing(10)
        if title:
            self.lay.addWidget(StrongBodyLabel(title))
        if description:
            d = CaptionLabel(description)
            d.setWordWrap(True)
            self.lay.addWidget(d)


def row(*widgets, stretch_last: bool = False, spacing: int = 8) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setSpacing(spacing)
    for w in widgets:
        if w is None:
            h.addStretch(1)
        elif isinstance(w, QWidget):
            h.addWidget(w)
        else:
            h.addLayout(w)
    if stretch_last:
        h.addStretch(1)
    return h


def labeled(label: str, widget: QWidget, width: int = 150) -> QHBoxLayout:
    lab = BodyLabel(label)
    lab.setFixedWidth(width)
    return row(lab, widget, stretch_last=False)


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: rgba(128,128,128,60);")
    return f


def zh_switch(sw):
    """SwitchButton with Chinese on/off text."""
    sw.setOnText("开")
    sw.setOffText("关")
    return sw


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(max(0, seconds))
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"

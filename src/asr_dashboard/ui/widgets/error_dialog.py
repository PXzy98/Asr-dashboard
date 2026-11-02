"""Error dialog with the full (redacted) traceback and a copy button."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QVBoxLayout
from qfluentwidgets import BodyLabel, PlainTextEdit, PrimaryPushButton, PushButton, SubtitleLabel

from asr_dashboard.redact import redact


class ErrorDialog(QDialog):
    def __init__(self, title: str, message: str, details: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 480)
        lay = QVBoxLayout(self)
        lay.addWidget(SubtitleLabel(title))
        msg = BodyLabel(redact(message))
        msg.setWordWrap(True)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(msg)
        self.details = PlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlainText(redact(details or message))
        lay.addWidget(self.details, 1)
        btns = QHBoxLayout()
        copy = PushButton("复制错误信息")
        copy.clicked.connect(self.copy)
        ok = PrimaryPushButton("关闭")
        ok.clicked.connect(self.accept)
        btns.addStretch(1)
        btns.addWidget(copy)
        btns.addWidget(ok)
        lay.addLayout(btns)

    def copy(self) -> None:
        QApplication.clipboard().setText(self.details.toPlainText())


def show_error(parent, title: str, message: str, details: str = "") -> None:
    ErrorDialog(title, message, details, parent).exec()

"""Builds a settings form from ``SettingField`` descriptions (used for engines and most settings sections)."""

from __future__ import annotations

import json
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QFormLayout, QHBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    DoubleSpinBox,
    LineEdit,
    PasswordLineEdit,
    PlainTextEdit,
    PushButton,
    SpinBox,
    SwitchButton,
    ToolTipFilter,
)

from asr_dashboard.settings_fields import SettingField, condition_met, get_dotted, set_dotted


class PathEdit(QWidget):
    changed = pyqtSignal()

    def __init__(self, directory: bool = True, parent=None) -> None:
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.edit = LineEdit()
        self.edit.textChanged.connect(self.changed)
        btn = PushButton("浏览…")
        btn.clicked.connect(self.browse)
        self.directory = directory
        h.addWidget(self.edit, 1)
        h.addWidget(btn)

    def browse(self) -> None:
        if self.directory:
            p = QFileDialog.getExistingDirectory(self, "选择目录", self.edit.text())
        else:
            p, _ = QFileDialog.getOpenFileName(self, "选择文件", self.edit.text())
        if p:
            self.edit.setText(p)

    def text(self) -> str:
        return self.edit.text()

    def setText(self, t: str) -> None:
        self.edit.setText(t)


class SchemaForm(QWidget):
    changed = pyqtSignal()

    def __init__(self, fields: list[SettingField], values: dict | None = None, parent=None,
                 disabled: dict[str, str] | None = None) -> None:
        super().__init__(parent)
        self.fields = fields
        self.disabled = disabled or {}  # key -> reason (availability)
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setVerticalSpacing(10)
        self.widgets: dict[str, QWidget] = {}
        self.labels: dict[str, QWidget] = {}
        for f in fields:
            w = self._make(f)
            self.widgets[f.key] = w
            lab = BodyLabel(f.label)
            if f.description:
                lab.setToolTip(f.description)
                lab.installEventFilter(ToolTipFilter(lab))
                w.setToolTip(f.description)
            self.labels[f.key] = lab
            container = QWidget()
            v = QHBoxLayout(container)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(w, 1)
            if f.description and f.type in ("bool", "int", "float"):
                cap = CaptionLabel(f.description.split("\n")[0][:60])
                cap.setWordWrap(False)
                v.addWidget(cap)
            self.form.addRow(lab, container)
            self.widgets[f.key + "__row"] = container
        self.set_values(values or {})
        self._refresh()

    # ------------------------------------------------------------------ widgets
    def _make(self, f: SettingField) -> QWidget:
        if f.type == "bool":
            w = SwitchButton()
            w.setOnText("开")
            w.setOffText("关")
            w.checkedChanged.connect(self._on_change)
        elif f.type == "int":
            w = SpinBox()
            lo = f.minimum if f.minimum is not None else -10**9
            if f.special_values:
                lo = min([lo] + list(f.special_values))
            w.setRange(int(lo), int(f.maximum if f.maximum is not None else 10**9))
            if f.special_values:
                k, v = next(iter(f.special_values.items()))
                if k == w.minimum():
                    w.setSpecialValueText(v)
            w.valueChanged.connect(self._on_change)
        elif f.type == "float":
            w = DoubleSpinBox()
            w.setDecimals(3 if (f.step or 1) < 0.1 else 2)
            w.setRange(f.minimum if f.minimum is not None else -1e9, f.maximum if f.maximum is not None else 1e9)
            if f.step:
                w.setSingleStep(f.step)
            w.valueChanged.connect(self._on_change)
        elif f.type == "enum":
            w = ComboBox()
            for i, (val, label) in enumerate(f.choices):
                w.addItem(label, userData=val)
                if val in f.disabled_choices:
                    w.setItemEnabled(i, False)
            w.currentIndexChanged.connect(self._on_change)
        elif f.type == "secret":
            w = PasswordLineEdit()
            w.textChanged.connect(self._on_change)
        elif f.type in ("text", "json"):
            w = PlainTextEdit()
            w.setFixedHeight(90)
            w.textChanged.connect(self._on_change)
        elif f.type == "path":
            w = PathEdit(directory=True)
            w.changed.connect(self._on_change)
        else:
            w = LineEdit()
            w.textChanged.connect(self._on_change)
        w.setMinimumWidth(260)
        return w

    def _on_change(self, *_a) -> None:
        self._refresh()
        self.changed.emit()

    # ------------------------------------------------------------------ values
    def set_values(self, values: dict) -> None:
        for f in self.fields:
            w = self.widgets[f.key]
            v = get_dotted(values, f.key, f.default)
            w.blockSignals(True)
            try:
                if f.type == "bool":
                    w.setChecked(bool(v))
                elif f.type == "int":
                    w.setValue(int(v if v is not None else 0))
                elif f.type == "float":
                    w.setValue(float(v if v is not None else 0.0))
                elif f.type == "enum":
                    idx = next((i for i, (val, _l) in enumerate(f.choices) if val == v), 0)
                    w.setCurrentIndex(idx)
                elif f.type == "json":
                    w.setPlainText(json.dumps(v if v is not None else {}, ensure_ascii=False, indent=1))
                elif f.type == "text":
                    w.setPlainText(str(v or ""))
                else:
                    w.setText(str(v or ""))
            finally:
                w.blockSignals(False)
        self._refresh()

    def _raw(self, f: SettingField) -> Any:
        w = self.widgets[f.key]
        if f.type == "bool":
            return w.isChecked()
        if f.type in ("int", "float"):
            return w.value()
        if f.type == "enum":
            return w.currentData()
        if f.type in ("text", "json"):
            return w.toPlainText()
        return w.text()

    def values(self) -> dict:
        """Validated values as a nested dict; raises ValueError with a readable message."""
        out: dict = {}
        for f in self.fields:
            try:
                set_dotted(out, f.key, f.validate(self._raw(f)))
            except json.JSONDecodeError as e:
                raise ValueError(f"{f.label}：JSON 格式错误（{e.msg}，第 {e.lineno} 行）") from e
            except (TypeError, ValueError) as e:
                raise ValueError(f"{f.label}：{e}") from e
        return out

    def flat_values(self) -> dict:
        return {f.key: self._raw(f) for f in self.fields}

    def _refresh(self) -> None:
        current: dict = {}
        for f in self.fields:
            try:
                set_dotted(current, f.key, self._raw(f))
            except Exception:  # noqa: BLE001
                pass
        for f in self.fields:
            visible = condition_met(f.visible_if, current)
            self.form.setRowVisible(self.widgets[f.key + "__row"], visible)
            enabled = condition_met(f.enabled_if, current) and f.key not in self.disabled
            self.widgets[f.key].setEnabled(enabled)
            if f.key in self.disabled:
                self.widgets[f.key].setToolTip(self.disabled[f.key])

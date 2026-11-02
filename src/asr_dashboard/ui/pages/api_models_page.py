"""API model management: add / edit / copy / delete, create from presets, connection test."""

from __future__ import annotations

import json

from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidgetItem
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    MessageBox,
    PlainTextEdit,
    PrimaryPushButton,
    PushButton,
    TableWidget,
)
from qfluentwidgets import FluentIcon as FIF

from asr_dashboard.config import secrets as sec
from asr_dashboard.config.schema import ApiModelConfig
from asr_dashboard.engines.api import PRESETS, PROVIDER_LABELS, create_api_engine, from_preset, provider_class
from asr_dashboard.ui.widgets.common import Card, Page, error, row, run_bg, success
from asr_dashboard.ui.widgets.schema_form import SchemaForm


class ApiModelsPage(Page):
    def __init__(self, services, on_changed=None, parent=None) -> None:
        super().__init__("apiModelsPage", "API 模型", "通过配置接入任意云端或自部署的识别服务。API Key 保存在系统钥匙串中，不写入 config.yaml。")
        self.services = services
        self.on_changed = on_changed
        top = Card()
        self.preset = ComboBox()
        self.preset.addItem("空白：OpenAI 兼容转写", userData="")
        for k, p in PRESETS.items():
            self.preset.addItem(f"预设：{p['name']}", userData=k)
        self.kind = ComboBox()
        for k, label in PROVIDER_LABELS.items():
            self.kind.addItem(label, userData=k)
        new = PrimaryPushButton(FIF.ADD, "新建")
        new.clicked.connect(self.create)
        copy = PushButton(FIF.COPY, "复制")
        copy.clicked.connect(self.duplicate)
        delete = PushButton(FIF.DELETE, "删除")
        delete.clicked.connect(self.delete)
        top.lay.addLayout(row(self.preset, BodyLabel("或接入类型"), self.kind, new, copy, delete, None))
        if services.secrets.is_plaintext:
            w = CaptionLabel(f"⚠ 系统钥匙串不可用，API Key 将以明文保存在 {services.secrets.file_path}")
            w.setStyleSheet("color:#c50")
            top.lay.addWidget(w)
        self.table = TableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["名称", "类型", "地址", "模型"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.table.setFixedHeight(200)
        top.lay.addWidget(self.table)
        self.body.addWidget(top)

        self.edit_card = Card("编辑")
        self.form_holder = self.edit_card.lay
        self.form: SchemaForm | None = None
        save = PrimaryPushButton(FIF.SAVE, "保存")
        save.clicked.connect(self.save)
        test = PushButton(FIF.SEND, "测试连接")
        test.clicked.connect(self.test)
        self.err = CaptionLabel("")
        self.err.setStyleSheet("color:#d13438")
        self.btn_row = row(save, test, self.err, None)
        self.edit_card.lay.addLayout(self.btn_row)
        self.body.addWidget(self.edit_card)

        res = Card("测试结果")
        self.result = PlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setFixedHeight(220)
        res.lay.addWidget(self.result)
        self.body.addWidget(res)
        self.models: list[ApiModelConfig] = []
        self.refresh()

    def refresh(self) -> None:
        self.models = list(self.services.cfg.api_models)
        self.table.setRowCount(len(self.models))
        for r, m in enumerate(self.models):
            for c, v in enumerate([m.name, PROVIDER_LABELS.get(m.type, m.type), m.base_url, m.model]):
                self.table.setItem(r, c, QTableWidgetItem(v))
        if self.models and self.table.currentRow() < 0:
            self.table.selectRow(0)
        self.show_selected()

    def _current(self) -> ApiModelConfig | None:
        r = self.table.currentRow()
        return self.models[r] if 0 <= r < len(self.models) else None

    def show_selected(self) -> None:
        m = self._current()
        if self.form is not None:
            self.form.setParent(None)
            self.form.deleteLater()
            self.form = None
        if not m:
            return
        values = m.model_dump(mode="json")
        values["api_key"] = self.services.secrets.get(sec.api_key_name(m.id))
        self.form = SchemaForm(provider_class(m.type).settings_schema(), values)
        self.edit_card.lay.insertWidget(self.edit_card.lay.count() - 1, self.form)
        self.err.setText("")

    def _changed(self) -> None:
        self.refresh()
        if self.on_changed:
            self.on_changed()

    def create(self) -> None:
        key = self.preset.currentData()
        m = from_preset(key) if key else ApiModelConfig(type=self.kind.currentData(), name="新 API 模型")
        self.services.config.update(lambda c: c.api_models.append(m))
        self._changed()
        self.table.selectRow(len(self.models) - 1)

    def duplicate(self) -> None:
        m = self._current()
        if not m:
            return
        data = m.model_dump()
        data.pop("id")
        data["name"] = m.name + "（副本）"
        new = ApiModelConfig(**data)
        key = self.services.secrets.get(sec.api_key_name(m.id))
        self.services.config.update(lambda c: c.api_models.append(new))
        if key:
            self.services.secrets.set(sec.api_key_name(new.id), key)
        self._changed()

    def delete(self) -> None:
        m = self._current()
        if not m or not MessageBox("删除 API 模型", f"确定删除 {m.name}？", self.window()).exec():
            return
        self.services.config.update(lambda c: c.api_models.remove(c.api_model(m.id)))
        self.services.secrets.delete(sec.api_key_name(m.id))
        self._changed()

    def _form_model(self) -> tuple[ApiModelConfig, str]:
        m = self._current()
        vals = self.form.values()
        key = vals.pop("api_key", "")
        data = m.model_dump()
        for k, v in vals.items():
            if k == "options" and isinstance(v, dict):
                data["options"] = {**data.get("options", {}), **v}
            else:
                data[k] = v
        return ApiModelConfig.model_validate(data), key

    def save(self) -> None:
        if not self._current() or not self.form:
            return
        try:
            new, key = self._form_model()
        except Exception as e:  # noqa: BLE001
            self.err.setText(str(e))
            return
        row_idx = self.table.currentRow()

        def mut(c):
            idx = next(i for i, x in enumerate(c.api_models) if x.id == new.id)
            c.api_models[idx] = new

        self.services.config.update(mut)
        self.services.secrets.set(sec.api_key_name(new.id), key)
        self._changed()
        self.table.selectRow(row_idx)
        success(self, "已保存", new.name)

    def test(self) -> None:
        if not self._current() or not self.form:
            return
        try:
            m, key = self._form_model()
        except Exception as e:  # noqa: BLE001
            self.err.setText(str(e))
            return
        self.result.setPlainText("正在用内置的 3 秒测试音频调用一次…")
        eng = create_api_engine(m, key, self.services.cfg.app.temp_dir or None)

        def ok(res):
            raw = res["raw"]
            raw_s = json.dumps(raw, ensure_ascii=False, indent=1) if not isinstance(raw, str) else raw
            seg_s = "\n".join(f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in res["segments"])
            self.result.setPlainText(f"耗时 {res['elapsed']:.2f} 秒 · 时间戳：{'有' if res['has_timestamps'] else '无'}\n\n"
                                     f"解析结果：\n{seg_s or '（空）'}\n\n原始响应：\n{raw_s[:4000]}")
            success(self, "连接成功", f"{res['elapsed']:.2f} 秒")

        def fail(msg, tb):
            self.result.setPlainText(f"失败：{msg}")
            error(self, "测试失败", msg[:200], 6000)

        run_bg(eng.test_connection, on_ok=ok, on_err=fail, parent=self)

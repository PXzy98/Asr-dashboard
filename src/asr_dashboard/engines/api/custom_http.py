"""Fully user-defined HTTP ASR service: request templates + JSONPath response extraction + optional polling."""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

from asr_dashboard.engines.api.base import APIError, BaseAPIEngine, CallResult, Upload
from asr_dashboard.models import CancelToken, Segment
from asr_dashboard.settings_fields import SettingField

_PH = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render_str(tpl: str, vars: dict[str, Any]) -> Any:
    m = _PH.fullmatch(tpl.strip()) if isinstance(tpl, str) else None
    if m and m.group(1) in vars:
        return vars[m.group(1)]  # whole-string placeholder keeps the raw value (e.g. numbers / None)
    return _PH.sub(lambda mm: "" if vars.get(mm.group(1)) is None else str(vars.get(mm.group(1), mm.group(0))), tpl)


def render(obj: Any, vars: dict[str, Any]) -> Any:
    if isinstance(obj, str):
        return render_str(obj, vars)
    if isinstance(obj, dict):
        return {render_str(k, vars) if isinstance(k, str) else k: render(v, vars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [render(v, vars) for v in obj]
    return obj


def jp_first(data: Any, path: str | None) -> Any:
    if not path:
        return None
    from jsonpath_ng.ext import parse

    found = parse(path).find(data)
    return found[0].value if found else None


def extract(data: Any, o: dict) -> tuple[str, list[Segment]]:
    scale = 0.001 if o.get("time_unit", "s") == "ms" else 1.0
    text = jp_first(data, o.get("text_path"))
    segs: list[Segment] = []
    arr = jp_first(data, o.get("segments_path")) if o.get("segments_path") else None
    if isinstance(arr, list):
        for item in arr:
            t = jp_first(item, o.get("seg_text", "$.text"))
            if t is None:
                continue
            st = jp_first(item, o.get("seg_start", "$.start"))
            en = jp_first(item, o.get("seg_end", "$.end"))
            spk = jp_first(item, o.get("seg_speaker")) if o.get("seg_speaker") else None
            segs.append(Segment(str(t).strip(), float(st or 0) * scale, float(en or 0) * scale,
                                speaker=None if spk is None else str(spk)))
    return ("" if text is None else str(text)), segs


class CustomHTTPEngine(BaseAPIEngine):
    provider_type = "custom_http"
    provider_label = "自定义 HTTP"

    @classmethod
    def provider_fields(cls) -> list[SettingField]:
        return [
            SettingField("options.method", "请求方法", "enum", "POST", choices=[("POST", "POST"), ("PUT", "PUT")]),
            SettingField("options.url", "URL 模板", "str", "{{base_url}}/transcribe",
                         "可用占位符：{{base_url}} {{api_key}} {{model}} {{language}} {{context}}"),
            SettingField("options.headers", "请求头模板（JSON）", "json", {"Authorization": "Bearer {{api_key}}"}),
            SettingField("options.body_type", "请求体类型", "enum", "multipart",
                         choices=[("multipart", "multipart 表单"), ("json_base64", "JSON + base64"),
                                  ("raw", "原始二进制")]),
            SettingField("options.file_field", "文件字段名（multipart）", "str", "file",
                         visible_if={"options.body_type": "multipart"}),
            SettingField("options.body", "请求体模板（JSON）", "json", {"model": "{{model}}", "language": "{{language}}"},
                         "multipart 时作为表单字段；JSON 时可用 {{audio_base64}} {{filename}} {{mime}}"),
            SettingField("options.text_path", "全文 JSONPath", "str", "$.text"),
            SettingField("options.segments_path", "分段数组 JSONPath", "str", "$.segments"),
            SettingField("options.seg_start", "分段 start 路径", "str", "$.start"),
            SettingField("options.seg_end", "分段 end 路径", "str", "$.end"),
            SettingField("options.seg_text", "分段 text 路径", "str", "$.text"),
            SettingField("options.seg_speaker", "分段 speaker 路径", "str", ""),
            SettingField("options.time_unit", "时间单位", "enum", "s", choices=[("s", "秒"), ("ms", "毫秒")]),
            SettingField("options.async", "异步模式（提交后轮询）", "bool", False),
            SettingField("options.task_id_path", "任务 id JSONPath", "str", "$.id", visible_if={"options.async": True}),
            SettingField("options.poll_url", "轮询 URL 模板", "str", "{{base_url}}/tasks/{{task_id}}",
                         visible_if={"options.async": True}),
            SettingField("options.status_path", "状态 JSONPath", "str", "$.status", visible_if={"options.async": True}),
            SettingField("options.done_values", "完成状态值（逗号分隔）", "str", "done,succeeded,SUCCEEDED,completed",
                         visible_if={"options.async": True}),
            SettingField("options.failed_values", "失败状态值（逗号分隔）", "str", "failed,FAILED,error",
                         visible_if={"options.async": True}),
            SettingField("options.result_path", "结果所在 JSONPath（空 = 轮询响应本身）", "str", "",
                         visible_if={"options.async": True}),
            SettingField("options.poll_interval", "轮询间隔（秒）", "float", 2.0, minimum=0.1, maximum=120,
                         visible_if={"options.async": True}),
            SettingField("options.max_wait", "最长等待（秒）", "float", 1800, minimum=1, maximum=86400,
                         visible_if={"options.async": True}),
        ]

    def provider_timestamps(self) -> bool:
        return bool(self.cfg.options.get("segments_path"))

    def vars(self, language, context, up: Upload | None = None, **extra) -> dict:
        v = {"base_url": self.cfg.base_url.rstrip("/"), "api_key": self.api_key, "model": self.cfg.model,
             "language": language or "", "context": context or ""}
        if up is not None:
            v.update(filename=up.filename, mime=up.mime, audio_base64=base64.b64encode(up.data).decode())
        v.update(extra)
        return v

    def call(self, up: Upload, language: str | None, context: str | None, cancel: CancelToken) -> CallResult:
        o = self.cfg.options
        v = self.vars(language, context, up)
        url = render_str(o.get("url", "{{base_url}}"), v)
        headers = {k: str(val) for k, val in (render(o.get("headers") or {}, v) or {}).items() if val not in (None, "")}
        body_tpl = {**(o.get("body") or {}), **(self.cfg.extra or {})}
        bt = o.get("body_type", "multipart")
        kw: dict = {"headers": headers}
        if bt == "multipart":
            fields = {k: ("" if val is None else str(val) if not isinstance(val, (dict, list)) else json.dumps(val))
                      for k, val in render(body_tpl, v).items()}
            kw["data"] = {k: val for k, val in fields.items() if val != ""}
            kw["files"] = {o.get("file_field", "file"): (up.filename, up.data, up.mime)}
        elif bt == "json_base64":
            kw["content"] = json.dumps(render(body_tpl, v))
            headers.setdefault("Content-Type", "application/json")
        else:
            kw["content"] = up.data
            headers.setdefault("Content-Type", up.mime)
        resp = self.request(o.get("method", "POST"), url, cancel, **kw)
        data = _json(resp.text)
        if o.get("async"):
            data = self.poll(data, v, cancel)
        text, segs = extract(data, o)
        if segs and any(s.end > 0 for s in segs):
            return CallResult(data, segs, True)
        if not text and segs:
            text = " ".join(s.text for s in segs)
        return CallResult(data, [Segment(text.strip(), 0.0, up.duration)] if text.strip() else [], False)

    def poll(self, submit: Any, v: dict, cancel: CancelToken) -> Any:
        o = self.cfg.options
        task_id = jp_first(submit, o.get("task_id_path", "$.id"))
        if task_id is None:
            raise APIError(f"提交响应中找不到任务 id（{o.get('task_id_path')}）：{json.dumps(submit)[:300]}", kind="parse")
        url = render_str(o.get("poll_url", ""), {**v, "task_id": task_id})
        headers = {k: str(val) for k, val in (render(o.get("headers") or {}, v) or {}).items() if val not in (None, "")}
        done = {x.strip() for x in str(o.get("done_values", "done")).split(",") if x.strip()}
        failed = {x.strip() for x in str(o.get("failed_values", "failed")).split(",") if x.strip()}
        deadline = time.time() + float(o.get("max_wait", 1800))
        while True:
            cancel.raise_if_cancelled()
            data = _json(self.request("GET", url, cancel, headers=headers).text)
            status = str(jp_first(data, o.get("status_path", "$.status")))
            if status in done:
                rp = o.get("result_path")
                return jp_first(data, rp) if rp else data
            if status in failed:
                raise APIError(f"任务失败（状态 {status}）：{json.dumps(data)[:300]}")
            if time.time() > deadline:
                raise APIError("轮询超时", kind="timeout")
            self._wait(float(o.get("poll_interval", 2.0)), cancel)


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}

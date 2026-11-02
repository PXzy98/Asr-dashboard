"""Alibaba Cloud Model Studio (DashScope) Qwen3-ASR recorded-file recognition (async FileTrans).

Verified from the dashscope 2.0.3 SDK source:
* submit  ``POST {host}/api/v1/services/audio/asr/transcription`` with header ``X-DashScope-Async: enable``
  and body ``{"model", "input": {...}, "parameters": {...}}``;
* poll    ``GET  {host}/api/v1/tasks/{task_id}`` until ``output.task_status`` is SUCCEEDED / FAILED;
* result  JSON downloaded from ``transcription_url``: ``transcripts[].sentences[]`` with
  ``begin_time``/``end_time`` in milliseconds;
* hosts: cn-beijing ``https://dashscope.aliyuncs.com``, international ``https://dashscope-intl.aliyuncs.com``.
FileTrans needs a URL: local audio is first uploaded to DashScope's free temporary storage
(``GET /api/v1/uploads?action=getPolicy&model=...`` -> OSS form upload -> ``oss://`` URL, valid 48 h), and
requests then need ``X-DashScope-OssResourceResolve: enable``. The temporary-storage flow and the exact
``input`` layout were confirmed via search results of the official API reference (``input.file_url``, one URL
per task; ``parameters.channel_id=[0]``, ``enable_itn``, ``enable_words``); both stay configurable
(``options.input_key``, ``options.file_url``).
"""

from __future__ import annotations

import json
import time
import uuid

from asr_dashboard.engines.api.base import APIError, BaseAPIEngine, CallResult, Upload
from asr_dashboard.models import CancelToken, Segment, Word
from asr_dashboard.settings_fields import SettingField

HOSTS = {"cn": "https://dashscope.aliyuncs.com", "intl": "https://dashscope-intl.aliyuncs.com"}


def parse_transcription(doc: dict) -> list[Segment]:
    segs: list[Segment] = []
    for tr in doc.get("transcripts") or []:
        for s in tr.get("sentences") or []:
            words = [Word(str(w.get("text", "")) + str(w.get("punctuation", "") or ""), w["begin_time"] / 1000,
                          w["end_time"] / 1000) for w in s.get("words") or [] if "begin_time" in w]
            segs.append(Segment(str(s.get("text", "")).strip(), s.get("begin_time", 0) / 1000,
                                s.get("end_time", 0) / 1000, speaker=_spk(s.get("speaker_id")), words=words))
        if not tr.get("sentences") and tr.get("text"):
            segs.append(Segment(tr["text"].strip(), 0.0, 0.0))
    return [s for s in segs if s.text]


def _spk(v) -> str | None:
    return None if v is None else f"SPEAKER_{int(v):02d}" if str(v).isdigit() else str(v)


class DashScopeQwenEngine(BaseAPIEngine):
    provider_type = "dashscope_qwen"
    provider_label = "阿里云百炼 Qwen3-ASR 录音文件识别"

    @classmethod
    def provider_fields(cls) -> list[SettingField]:
        return [
            SettingField("options.region", "地域", "enum", "cn", "", choices=[("cn", "中国大陆（北京）"),
                                                                        ("intl", "国际站（新加坡）")]),
            SettingField("options.input_key", "input 字段", "enum", "file_url",
                         "qwen3-asr-flash-filetrans 使用 file_url；paraformer / fun-asr 使用 file_urls",
                         choices=[("file_url", "file_url"), ("file_urls", "file_urls")]),
            SettingField("options.poll_interval", "轮询间隔（秒）", "float", 2.0, minimum=0.2, maximum=60),
            SettingField("options.max_wait", "最长等待（秒）", "float", 3600, minimum=10, maximum=86400),
        ]

    def host(self) -> str:
        if self.cfg.base_url:
            return self.cfg.base_url.rstrip("/").removesuffix("/api/v1")
        return HOSTS.get(self.cfg.options.get("region", "cn"), HOSTS["cn"])

    def wants_chunked_input(self) -> bool:
        return False  # returns sentence timestamps for the whole file

    def upload_temp(self, up: Upload, cancel: CancelToken) -> str:
        h = self.host()
        resp = self.request("GET", f"{h}/api/v1/uploads", cancel, headers=self.auth_headers(),
                            params={"action": "getPolicy", "model": self.cfg.model})
        try:
            pol = resp.json()["data"]
        except Exception as e:  # noqa: BLE001
            raise APIError(f"获取上传凭证失败：{resp.text[:300]}", kind="parse") from e
        key = f"{pol['upload_dir']}/{uuid.uuid4().hex}_{up.filename}"
        form = {"OSSAccessKeyId": pol["oss_access_key_id"], "Signature": pol["signature"], "policy": pol["policy"],
                "x-oss-object-acl": pol.get("x_oss_object_acl", "private"),
                "x-oss-forbid-overwrite": pol.get("x_oss_forbid_overwrite", "true"), "key": key,
                "success_action_status": "200"}
        self.request("POST", pol["upload_host"], cancel, data=form, files={"file": (up.filename, up.data, up.mime)})
        return f"oss://{key}"

    def call(self, up: Upload, language: str | None, context: str | None, cancel: CancelToken) -> CallResult:
        h = self.host()
        file_url = self.cfg.options.get("file_url") or self.upload_temp(up, cancel)
        params: dict = {"channel_id": [0], "enable_itn": False, "enable_words": True}
        if language:
            params["language"] = language
        if context:
            params["corpus"] = {"text": context}
        params.update(self.cfg.extra or {})
        key = self.cfg.options.get("input_key", "file_url")
        body = {"model": self.cfg.model, "input": {key: [file_url] if key == "file_urls" else file_url},
                "parameters": params}
        headers = {**self.auth_headers(), "Content-Type": "application/json", "X-DashScope-Async": "enable"}
        if file_url.startswith("oss://"):
            headers["X-DashScope-OssResourceResolve"] = "enable"
        resp = self.request("POST", f"{h}/api/v1/services/audio/asr/transcription", cancel, headers=headers,
                            content=json.dumps(body))
        task_id = (resp.json().get("output") or {}).get("task_id")
        if not task_id:
            raise APIError(f"提交任务失败：{resp.text[:300]}", kind="parse")
        deadline = time.time() + float(self.cfg.options.get("max_wait", 3600))
        interval = float(self.cfg.options.get("poll_interval", 2.0))
        while True:
            cancel.raise_if_cancelled()
            r = self.request("GET", f"{h}/api/v1/tasks/{task_id}", cancel, headers=self.auth_headers())
            out = r.json().get("output") or {}
            status = out.get("task_status")
            if status == "SUCCEEDED":
                break
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise APIError(f"DashScope 任务失败：{out.get('message') or out.get('code') or r.text[:300]}")
            if time.time() > deadline:
                raise APIError("DashScope 任务等待超时", kind="timeout")
            self._wait(interval, cancel)
        url = (out.get("result") or {}).get("transcription_url")
        if not url:
            results = out.get("results") or []
            url = next((x.get("transcription_url") for x in results if x.get("transcription_url")), None)
        if not url:
            raise APIError(f"任务成功但没有 transcription_url：{json.dumps(out)[:300]}", kind="parse")
        doc = self.request("GET", url, cancel).json()  # presigned URL, no auth header
        segs = parse_transcription(doc)
        return CallResult(doc, segs, any(s.end > 0 for s in segs), language)

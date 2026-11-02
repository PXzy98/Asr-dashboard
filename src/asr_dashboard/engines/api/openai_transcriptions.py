"""``POST {base_url}/audio/transcriptions`` (OpenAI, Groq, faster-whisper-server, whisper.cpp, GLM-ASR, Azure).

Fields: model, language, prompt (context), temperature, response_format, timestamp_granularities[].
``verbose_json`` is requested when timestamps are wanted; if the server rejects it we fall back to
``json`` and remember that this model has no timestamps.
"""

from __future__ import annotations

import json
import re

from asr_dashboard.engines.api.base import APIError, BaseAPIEngine, CallResult, Upload, words_into_segments
from asr_dashboard.models import CancelToken, Segment, Word
from asr_dashboard.settings_fields import SettingField

_LANG_NAMES = {"english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko", "french": "fr", "german": "de",
               "spanish": "es", "russian": "ru", "portuguese": "pt", "italian": "it", "cantonese": "yue"}


def parse_srt_vtt(text: str) -> list[Segment]:
    segs = []
    pat = re.compile(r"(\d+:)?(\d{1,2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+:)?(\d{1,2}):(\d{2})[,.](\d{3})[^\n]*\n(.*?)(?:\n\s*\n|\Z)",
                     re.S)

    def t(h, m, s, ms):
        return int((h or "0:")[:-1]) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    for g in pat.finditer(text.replace("\r\n", "\n")):
        body = g.group(9).strip()
        if body:
            segs.append(Segment(body, t(*g.group(1, 2, 3, 4)), t(*g.group(5, 6, 7, 8))))
    return segs


class OpenAITranscriptionsEngine(BaseAPIEngine):
    provider_type = "openai_transcriptions"
    provider_label = "OpenAI 兼容转写接口"

    @classmethod
    def provider_fields(cls) -> list[SettingField]:
        return [
            SettingField("options.response_format", "返回格式", "enum", "verbose_json",
                         "verbose_json 带时间戳；不支持时自动降级为 json",
                         choices=[(x, x) for x in ("verbose_json", "json", "text", "srt", "vtt")]),
            SettingField("options.temperature", "temperature", "float", 0.0, minimum=0, maximum=1),
            SettingField("options.word_timestamps", "请求词级时间戳", "bool", False,
                         "timestamp_granularities[]=word（服务端支持时）"),
            SettingField("options.azure", "Azure OpenAI", "bool", False,
                         "base_url 填 https://<资源名>.openai.azure.com，认证头使用 api-key"),
            SettingField("options.deployment", "Azure 部署名", "str", "", visible_if={"options.azure": True}),
            SettingField("options.api_version", "Azure api-version", "str", "2024-06-01",
                         visible_if={"options.azure": True}),
        ]

    def url(self) -> str:
        o = self.cfg.options
        base = self.cfg.base_url.rstrip("/")
        if o.get("azure"):
            return f"{base}/openai/deployments/{o.get('deployment', '')}/audio/transcriptions?api-version={o.get('api_version', '2024-06-01')}"
        return f"{base}/audio/transcriptions"

    def auth_headers(self) -> dict[str, str]:
        if self.cfg.options.get("azure"):
            return {"api-key": self.api_key} if self.api_key else {}
        return super().auth_headers()

    def _format(self) -> str:
        fmt = self.cfg.options.get("response_format", "verbose_json")
        if not self._timestamps_supported and fmt == "verbose_json":
            return "json"
        return fmt

    def call(self, up: Upload, language: str | None, context: str | None, cancel: CancelToken) -> CallResult:
        fmt = self._format()
        try:
            return self._post(up, language, context, fmt, cancel)
        except APIError as e:
            if fmt == "verbose_json" and e.status in (400, 422) and re.search(r"response_format|verbose|timestamp", e.body or str(e), re.I):
                self._timestamps_supported = False
                self.warnings.append(f"{self.cfg.name} 不支持 verbose_json，已改用 json（无时间戳，改为分段请求）")
                return self._post(up, language, context, "json", cancel)
            raise

    def _post(self, up: Upload, language, context, fmt: str, cancel: CancelToken) -> CallResult:
        data: list[tuple[str, str]] = []
        if self.cfg.model:
            data.append(("model", self.cfg.model))
        if language:
            data.append(("language", language))
        if context:
            data.append(("prompt", context))
        data.append(("response_format", fmt))
        temp = self.cfg.options.get("temperature")
        if temp is not None:
            data.append(("temperature", str(temp)))
        if fmt == "verbose_json":
            data.append(("timestamp_granularities[]", "segment"))
            if self.cfg.options.get("word_timestamps"):
                data.append(("timestamp_granularities[]", "word"))
        for k, v in (self.cfg.extra or {}).items():
            for item in v if isinstance(v, list) else [v]:
                data.append((k, item if isinstance(item, str) else json.dumps(item)))
        form: dict[str, str | list[str]] = {}
        for k, val in data:  # httpx wants repeated fields as lists
            if k in form:
                prev = form[k]
                form[k] = (prev if isinstance(prev, list) else [prev]) + [val]
            else:
                form[k] = val
        files = {"file": (up.filename, up.data, up.mime)}
        resp = self.request("POST", self.url(), cancel, headers=self.auth_headers(), data=form, files=files)
        return self.parse(resp.text, fmt, up.duration)

    def parse(self, body: str, fmt: str, duration: float) -> CallResult:
        if fmt in ("srt", "vtt"):
            segs = parse_srt_vtt(body)
            return CallResult(body, segs, bool(segs))
        if fmt == "text":
            return CallResult(body, [Segment(body.strip(), 0.0, duration)], False)
        try:
            d = json.loads(body)
        except json.JSONDecodeError as e:
            raise APIError(f"无法解析返回内容：{body[:200]}", kind="parse") from e
        lang = d.get("language")
        if isinstance(lang, str):
            lang = _LANG_NAMES.get(lang.lower(), lang.lower()[:3] if len(lang) > 3 else lang.lower())
        segs_raw = d.get("segments") or []
        if segs_raw:
            segs = [Segment(str(s.get("text", "")).strip(), float(s.get("start", 0)), float(s.get("end", 0)),
                            speaker=s.get("speaker"), language=lang) for s in segs_raw]
            words = [Word(str(w.get("word", w.get("text", ""))).strip(), float(w["start"]), float(w["end"]))
                     for w in d.get("words") or [] if "start" in w]
            if words:
                words_into_segments(segs, words)
            return CallResult(d, [s for s in segs if s.text], True, lang)
        words = d.get("words") or []
        if words and all("start" in w for w in words):
            from asr_dashboard.core.sentences import split_sentences

            ws = [Word(str(w.get("word", "")).strip(), float(w["start"]), float(w["end"])) for w in words]
            return CallResult(d, split_sentences(ws), True, lang)
        return CallResult(d, [Segment(str(d.get("text", "")).strip(), 0.0, duration, language=lang)], False, lang)

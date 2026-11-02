"""pydantic v2 configuration model.

Loading never fails: invalid fields are replaced by their defaults and reported (see ``load_config``).
Secrets (API keys, HF token) are NOT part of this model; they live in the secret store (config/secrets.py).
"""

from __future__ import annotations

import copy
import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

log = logging.getLogger(__name__)

CONFIG_VERSION = 1

QWEN_LANGUAGES: dict[str, str] = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino",
    "fa": "Persian", "el": "Greek", "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian",
}
ALIGNER_LANGUAGES = {
    "Chinese", "English", "Cantonese", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
}


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class AppSection(_Base):
    output_dir: str = "output"
    temp_dir: str = ""
    cleanup_temp: bool = True
    theme: Literal["system", "light", "dark"] = "system"
    ui_language: Literal["zh", "en"] = "zh"


class EngineSection(_Base):
    default: str = "qwen3-local"


class AlignerSection(_Base):
    enabled: bool = True
    device: str = "follow"
    dtype: Literal["follow", "auto", "bfloat16", "float16", "float32"] = "follow"
    attn_implementation: Literal["follow", "auto", "sdpa", "eager", "flash_attention_2"] = "follow"


class Qwen3Section(_Base):
    variant: Literal["1.7b", "0.6b"] = "1.7b"
    backend: Literal["qwen_asr", "transformers", "vllm"] = "qwen_asr"
    source: Literal["huggingface", "modelscope", "local"] = "huggingface"
    local_path: str = ""
    aligner_local_path: str = ""
    cache_dir: str = ""
    device: str = "auto"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    attn_implementation: Literal["auto", "sdpa", "eager", "flash_attention_2"] = "auto"
    torch_compile: bool = False
    max_inference_batch_size: int = 32
    max_new_tokens: int = Field(1024, ge=64, le=8192)
    gpu_memory_utilization: float = Field(0.7, ge=0.1, le=0.95)
    language: str = "auto"
    context: str = ""
    context_max_chars: int = Field(2000, ge=100, le=20000)
    keep_loaded: bool = True
    oom_fallback_cpu: bool = True
    aligner: AlignerSection = Field(default_factory=AlignerSection)

    @field_validator("max_inference_batch_size")
    @classmethod
    def _batch(cls, v: int) -> int:
        if v == -1 or 1 <= v <= 256:
            return v
        raise ValueError("max_inference_batch_size must be -1 or 1..256")

    @field_validator("language")
    @classmethod
    def _lang(cls, v: str) -> str:
        if v == "auto" or v in QWEN_LANGUAGES:
            return v
        raise ValueError(f"unsupported language {v}")

    @field_validator("device")
    @classmethod
    def _device(cls, v: str) -> str:
        if v in ("auto", "cpu") or (v.startswith("cuda:") and v[5:].isdigit()):
            return v
        raise ValueError(f"invalid device {v}")


class Glossary(_Base):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "词库"
    description: str = ""
    terms: list[str] = Field(default_factory=list)


class SegmentationSection(_Base):
    strategy: Literal["duration", "vad", "hybrid"] = "hybrid"
    target_duration: float = Field(60, ge=5, le=1200)
    min_segment: float = Field(5, ge=0.5, le=600)
    max_segment: float = Field(60, ge=5, le=1200)
    overlap: float = Field(0.5, ge=0, le=10)
    silence_threshold_db: float = Field(-40, ge=-90, le=0)
    min_silence: float = Field(0.3, ge=0.05, le=5)
    max_sentence_duration: float = Field(15, ge=2, le=120)
    search_window: float = Field(10, ge=0, le=60)

    @model_validator(mode="after")
    def _order(self) -> SegmentationSection:
        if self.min_segment > self.max_segment:
            raise ValueError("min_segment must be <= max_segment")
        return self


class PyannoteSection(_Base):
    model_dir: str = ""
    model_id: str = "pyannote/speaker-diarization-community-1"


class DiarizenSection(_Base):
    model_id: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    env_dir: str = ""
    license_accepted: bool = False


class EmbeddingDiarSection(_Base):
    model: str = "microsoft/wavlm-base-plus-sv"
    clustering: Literal["agglomerative", "spectral"] = "agglomerative"
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    window: float = Field(1.5, ge=0.5, le=10)
    step: float = Field(0.75, ge=0.1, le=10)


class DiarizationSection(_Base):
    enabled: bool = False
    backend: Literal["auto", "pyannote", "diarizen", "embedding"] = "auto"
    num_speakers: int = Field(0, ge=0, le=100)
    min_speakers: int = Field(1, ge=1, le=100)
    max_speakers: int = Field(20, ge=1, le=100)
    device: str = "follow"
    min_turn_duration: float = Field(0.3, ge=0, le=10)
    merge_gap_threshold: float = Field(0.5, ge=0, le=10)
    pyannote: PyannoteSection = Field(default_factory=PyannoteSection)
    diarizen: DiarizenSection = Field(default_factory=DiarizenSection)
    embedding: EmbeddingDiarSection = Field(default_factory=EmbeddingDiarSection)


class VoiceprintSection(_Base):
    enabled: bool = True
    embedding_model: Literal["pyannote", "wavlm"] = "pyannote"
    match_threshold: float = Field(0.6, ge=0.0, le=1.0)
    auto_add_on_correction: bool = False


class TextSection(_Base):
    merge_sentences: bool = True
    normalize_punctuation: bool = True
    clean_whitespace: bool = True
    remove_fillers: bool = False
    smart_capitalize: bool = True
    filler_words: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "zh": ["嗯", "啊", "呃", "那个", "就是"],
            "en": ["um", "uh", "you know"],
            "ja": ["えーと", "あの"],
        }
    )


class OutputSection(_Base):
    formats: list[Literal["txt", "srt", "vtt", "json", "ass", "md"]] = Field(default_factory=lambda: ["srt", "txt"])
    auto_save: bool = True
    streaming: bool = False
    include_speakers: bool = True
    txt_timestamps: bool = True
    ass_font: str = "Arial"
    ass_font_size: int = Field(20, ge=6, le=200)
    on_existing: Literal["skip", "overwrite", "rename"] = "skip"


class BatchSection(_Base):
    recursive: bool = True
    sort: Literal["natural", "mtime", "size"] = "natural"
    skip_hidden: bool = True
    skip_empty: bool = True
    extensions: list[str] = Field(default_factory=list)  # empty = all supported
    prevent_sleep: bool = True
    oom_retry: bool = True


class LLMTasks(_Base):
    cleanup: bool = True
    summary: bool = False
    speaker_guess: bool = False


class LLMSection(_Base):
    enabled: bool = True
    base_url: str = ""
    model: str = ""
    validated: bool = False
    json_mode: bool = False
    timeout: float = Field(120, ge=5, le=1800)
    temperature: float = Field(0.2, ge=0, le=2)
    chunk_chars: int = Field(1500, ge=200, le=20000)
    concurrency: int = Field(2, ge=1, le=16)
    tasks: LLMTasks = Field(default_factory=LLMTasks)
    max_change_ratio: float = Field(0.4, ge=0.0, le=1.0)
    privacy_acknowledged: bool = False
    context_chars: int = Field(24000, ge=1000, le=1_000_000)  # rough model context budget for summaries


class RealtimeSection(_Base):
    enabled: bool = True
    protocol: Literal["vllm_realtime", "dashscope_realtime"] = "vllm_realtime"
    url: str = ""
    model: str = ""
    language: str = "auto"
    validated: bool = False
    chunk_ms: int = Field(100, ge=20, le=2000)
    auto_reconnect: bool = True
    max_reconnects: int = Field(5, ge=0, le=100)
    refine_after_stop: bool = True
    live_speaker_id: bool = False
    source: Literal["microphone", "system", "mixed"] = "microphone"
    mic_device: str = ""
    system_device: str = ""
    glossary_ids: list[str] = Field(default_factory=list)


class ApiModelConfig(_Base):
    id: str = Field(default_factory=lambda: "api-" + uuid.uuid4().hex[:8])
    name: str = "API 模型"
    type: Literal["openai_transcriptions", "openai_chat_audio", "dashscope_qwen", "custom_http"] = (
        "openai_transcriptions"
    )
    preset: str = ""
    base_url: str = ""
    model: str = ""
    language: str = "auto"
    supports_context: bool = True
    timeout: float = Field(120, ge=1, le=7200)
    retries: int = Field(3, ge=0, le=20)
    concurrency: int = Field(2, ge=1, le=32)
    max_file_mb: float = Field(0, ge=0)
    max_audio_seconds: float = Field(0, ge=0)
    timestamps: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)
    # type-specific options (custom_http templates, azure deployment, dashscope region...)
    options: dict[str, Any] = Field(default_factory=dict)


class LoggingSection(_Base):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str = ""
    max_file_size_mb: int = Field(10, ge=1, le=1000)
    backup_count: int = Field(5, ge=0, le=100)


class AppConfig(_Base):
    version: int = CONFIG_VERSION
    app: AppSection = Field(default_factory=AppSection)
    engine: EngineSection = Field(default_factory=EngineSection)
    qwen3: Qwen3Section = Field(default_factory=Qwen3Section)
    glossaries: list[Glossary] = Field(default_factory=list)
    segmentation: SegmentationSection = Field(default_factory=SegmentationSection)
    diarization: DiarizationSection = Field(default_factory=DiarizationSection)
    voiceprint: VoiceprintSection = Field(default_factory=VoiceprintSection)
    text: TextSection = Field(default_factory=TextSection)
    output: OutputSection = Field(default_factory=OutputSection)
    batch: BatchSection = Field(default_factory=BatchSection)
    llm: LLMSection = Field(default_factory=LLMSection)
    realtime: RealtimeSection = Field(default_factory=RealtimeSection)
    api_models: list[ApiModelConfig] = Field(default_factory=list)
    logging: LoggingSection = Field(default_factory=LoggingSection)

    # ---- convenience --------------------------------------------------
    def api_model(self, model_id: str) -> ApiModelConfig | None:
        return next((m for m in self.api_models if m.id == model_id), None)

    def glossary(self, gid: str) -> Glossary | None:
        return next((g for g in self.glossaries if g.id == gid), None)

    @property
    def llm_active(self) -> bool:
        return bool(self.llm.enabled and self.llm.validated and self.llm.base_url and self.llm.model)

    @property
    def realtime_active(self) -> bool:
        return bool(self.realtime.enabled and self.realtime.validated and self.realtime.url)


def _delete_path(data: Any, loc: tuple) -> bool:
    """Delete the value at ``loc`` (pydantic error location) so the default is used instead."""
    cur = data
    for i, part in enumerate(loc):
        last = i == len(loc) - 1
        if isinstance(cur, dict):
            if part not in cur:
                return False
            if last:
                del cur[part]
                return True
            cur = cur[part]
        elif isinstance(cur, list) and isinstance(part, int):
            if part >= len(cur):
                return False
            if last or not isinstance(cur[part], (dict, list)):
                del cur[part]
                return True
            cur = cur[part]
        else:
            return False
    return False


def load_config(data: Any) -> tuple[AppConfig, list[str]]:
    """Validate ``data``; invalid values fall back to defaults. Returns (config, problems)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        if data is not None:
            problems.append("配置文件内容不是映射（mapping），已使用全部默认值")
        return AppConfig(), problems
    work = copy.deepcopy(data)
    for _ in range(200):
        try:
            cfg = AppConfig.model_validate(work)
            break
        except ValidationError as e:
            progressed = False
            for err in e.errors():
                loc = tuple(err.get("loc", ()))
                # model-level validators report loc of the section; drop the whole section then
                for cut in range(len(loc), 0, -1):
                    if _delete_path(work, loc[:cut]):
                        problems.append(f"配置项 {'.'.join(map(str, loc[:cut]))} 无效（{err.get('msg')}），已使用默认值")
                        progressed = True
                        break
            if not progressed:
                problems.append(f"配置无法解析，已使用全部默认值：{e}")
                cfg = AppConfig()
                break
    else:  # pragma: no cover - defensive
        cfg = AppConfig()
    if work.get("version", CONFIG_VERSION) != CONFIG_VERSION:
        problems.append(f"配置版本 {work.get('version')} 与当前版本 {CONFIG_VERSION} 不同，已按当前格式读取")
    for p in problems:
        log.warning(p)
    return cfg, problems

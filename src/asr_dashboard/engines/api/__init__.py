"""API engine providers and built-in presets (presets only pre-fill URL / fields; users add their own key)."""

from __future__ import annotations

from asr_dashboard.config.schema import ApiModelConfig


def provider_class(kind: str):
    if kind == "openai_transcriptions":
        from asr_dashboard.engines.api.openai_transcriptions import OpenAITranscriptionsEngine

        return OpenAITranscriptionsEngine
    if kind == "openai_chat_audio":
        from asr_dashboard.engines.api.openai_chat_audio import OpenAIChatAudioEngine

        return OpenAIChatAudioEngine
    if kind == "dashscope_qwen":
        from asr_dashboard.engines.api.dashscope_qwen import DashScopeQwenEngine

        return DashScopeQwenEngine
    if kind == "custom_http":
        from asr_dashboard.engines.api.custom_http import CustomHTTPEngine

        return CustomHTTPEngine
    raise KeyError(kind)


def create_api_engine(cfg: ApiModelConfig, api_key: str, temp_dir: str | None = None, transport=None):
    return provider_class(cfg.type)(cfg, api_key, temp_dir, transport)


PROVIDER_LABELS = {
    "openai_transcriptions": "OpenAI 兼容转写接口（/audio/transcriptions）",
    "openai_chat_audio": "OpenAI 兼容 Chat 音频输入（远程 Qwen3-ASR）",
    "dashscope_qwen": "阿里云百炼 Qwen3-ASR 录音文件识别",
    "custom_http": "自定义 HTTP",
}

# Values checked against public docs where reachable (see README "API 预设核实情况").
PRESETS: dict[str, dict] = {
    "openai": dict(name="OpenAI Whisper", type="openai_transcriptions", base_url="https://api.openai.com/v1",
                   model="whisper-1", max_file_mb=25, timestamps=True,
                   options={"response_format": "verbose_json"}),
    "openai-gpt4o": dict(name="OpenAI gpt-4o-transcribe", type="openai_transcriptions",
                         base_url="https://api.openai.com/v1", model="gpt-4o-transcribe", max_file_mb=25,
                         timestamps=False, options={"response_format": "json"}),
    "groq": dict(name="Groq Whisper", type="openai_transcriptions", base_url="https://api.groq.com/openai/v1",
                 model="whisper-large-v3-turbo", max_file_mb=25, timestamps=True,
                 options={"response_format": "verbose_json"}),
    "azure": dict(name="Azure OpenAI Whisper", type="openai_transcriptions",
                  base_url="https://<资源名>.openai.azure.com", model="", max_file_mb=25, timestamps=True,
                  options={"azure": True, "deployment": "whisper", "api_version": "2024-06-01",
                           "response_format": "verbose_json"}),
    "dashscope": dict(name="阿里云百炼 Qwen3-ASR", type="dashscope_qwen", base_url="",
                      model="qwen3-asr-flash-filetrans", timestamps=True, supports_context=False, concurrency=2,
                      timeout=300, options={"region": "cn", "input_key": "file_url"}),
    "vllm-qwen3": dict(name="自部署 vLLM Qwen3-ASR", type="openai_chat_audio", base_url="http://localhost:8000/v1",
                       model="Qwen/Qwen3-ASR-1.7B", timestamps=False, concurrency=4, timeout=300),
    "glm": dict(name="智谱 GLM-ASR", type="openai_transcriptions", base_url="https://open.bigmodel.cn/api/paas/v4",
                model="glm-asr-2512", timestamps=False, max_file_mb=25, max_audio_seconds=30,
                supports_context=True, options={"response_format": "json"}),
}


def from_preset(key: str) -> ApiModelConfig:
    return ApiModelConfig(preset=key, **PRESETS[key])

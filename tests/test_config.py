import yaml

from asr_dashboard.config.manager import ConfigManager
from asr_dashboard.config.schema import AppConfig, load_config
from asr_dashboard.config.secrets import SecretStore, api_key_name


def test_defaults_match_spec():
    c = AppConfig()
    assert c.version == 1
    assert c.engine.default == "qwen3-local"
    assert c.qwen3.variant == "1.7b" and c.qwen3.backend == "qwen_asr"
    assert c.qwen3.max_inference_batch_size == 32 and c.qwen3.max_new_tokens == 1024
    assert c.qwen3.aligner.enabled and c.qwen3.aligner.device == "follow"
    assert c.segmentation.strategy == "hybrid" and c.segmentation.max_segment == 60
    assert c.diarization.max_speakers == 20 and c.diarization.backend == "auto"
    assert c.voiceprint.match_threshold == 0.6
    assert c.llm.max_change_ratio == 0.4 and not c.llm.validated
    assert c.realtime.protocol == "vllm_realtime" and c.realtime.chunk_ms == 100
    assert c.output.formats == ["srt", "txt"]
    assert not c.llm_active and not c.realtime_active


def test_invalid_fields_fall_back_to_defaults():
    data = {
        "qwen3": {"variant": "9b", "max_new_tokens": 5, "language": "xx", "device": "cuda:1", "context": "abc"},
        "segmentation": {"min_segment": 100, "max_segment": 10},
        "diarization": {"num_speakers": "many"},
        "api_models": [{"id": "a", "type": "nope"}, {"id": "b", "name": "ok"}],
        "unknown_section": {"x": 1},
    }
    cfg, problems = load_config(data)
    assert cfg.qwen3.variant == "1.7b"
    assert cfg.qwen3.max_new_tokens == 1024
    assert cfg.qwen3.language == "auto"
    assert cfg.qwen3.device == "cuda:1"  # valid values are kept
    assert cfg.qwen3.context == "abc"
    assert cfg.segmentation.min_segment == 5 and cfg.segmentation.max_segment == 60
    assert cfg.diarization.num_speakers == 0
    assert [m.id for m in cfg.api_models] == ["a", "b"]
    assert cfg.api_models[0].type == "openai_transcriptions"
    assert len(problems) >= 5


def test_non_mapping_and_broken_yaml(tmp_path):
    cfg, problems = load_config(["not", "a", "dict"])
    assert isinstance(cfg, AppConfig) and problems
    p = tmp_path / "c.yaml"
    p.write_text("qwen3: [unclosed", encoding="utf-8")
    m = ConfigManager(p, SecretStore(tmp_path / "s.yaml", use_keyring=False))
    assert m.config == AppConfig() and m.problems


def test_roundtrip_and_no_secrets_in_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    store = SecretStore(tmp_path / "secrets.yaml", use_keyring=False)
    m = ConfigManager(p, store)
    m.update(lambda c: setattr(c.qwen3, "max_new_tokens", 2048))
    store.set(api_key_name("x"), "sk-supersecretvalue123")
    m2 = ConfigManager(p, store)
    assert m2.config.qwen3.max_new_tokens == 2048
    assert "sk-supersecret" not in p.read_text(encoding="utf-8")
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["qwen3"]["max_new_tokens"] == 2048
    assert store.get(api_key_name("x")) == "sk-supersecretvalue123"
    assert store.is_plaintext
    store.delete(api_key_name("x"))
    assert store.get(api_key_name("x")) == ""


def test_example_config_is_valid():
    from pathlib import Path

    ex = Path(__file__).resolve().parents[1] / "config.example.yaml"
    cfg, problems = load_config(yaml.safe_load(ex.read_text(encoding="utf-8")))
    assert problems == []

import pytest
from pathlib import Path
from fleet.config import Config, load_config, ModelEntry

SAMPLE_YAML = """
ollama:
  base_url: http://localhost:11434
models:
  deepseek-v4-pro:
    tags: [code, reasoning, math]
    priority: 1
  glm-5.1:
    tags: [creative, chinese, long_context]
    priority: 2
thresholds:
  single_confidence: 0.8
  parallel_timeout: 60
  max_parallel: 3
classifier:
  embeddings_model: all-MiniLM-L6-v2
"""

def test_load_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(SAMPLE_YAML)
    cfg = load_config(config_path)
    assert cfg.ollama.base_url == "http://localhost:11434"
    assert "deepseek-v4-pro" in cfg.models
    assert cfg.models["deepseek-v4-pro"].tags == ["code", "reasoning", "math"]
    assert cfg.models["deepseek-v4-pro"].priority == 1
    assert "glm-5.1" in cfg.models
    assert cfg.models["glm-5.1"].tags == ["creative", "chinese", "long_context"]
    assert cfg.models["glm-5.1"].priority == 2
    assert cfg.thresholds.single_confidence == 0.8
    assert cfg.thresholds.parallel_timeout == 60
    assert cfg.thresholds.max_parallel == 3
    assert cfg.classifier.embeddings_model == "all-MiniLM-L6-v2"


def test_cloud_suffix_stripping(tmp_path):
    yaml_text = """
models:
  deepseek-v4-pro:cloud:
    tags: [code]
    priority: 5
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)
    assert "deepseek-v4-pro" in cfg.models
    assert "deepseek-v4-pro:cloud" not in cfg.models
    assert cfg.models["deepseek-v4-pro"].tags == ["code"]
    assert cfg.models["deepseek-v4-pro"].priority == 5


def test_missing_file_fallback():
    nonexistent = Path("/nonexistent/path/config.yaml")
    cfg = load_config(nonexistent)
    assert isinstance(cfg, Config)
    assert cfg.ollama.base_url == "http://localhost:11434"
    assert cfg.models == {}
    assert cfg.thresholds.single_confidence == 0.8
    assert cfg.thresholds.parallel_timeout == 60
    assert cfg.thresholds.max_parallel == 3
    assert cfg.classifier.embeddings_model == "all-MiniLM-L6-v2"


def test_model_class_parses(tmp_path):
    yaml_text = """
models:
  thinker:
    tags: [reasoning]
    priority: 1
    class: reasoning
  chatter:
    tags: [creative]
    priority: 2
    class: chat
  default-class:
    tags: [code]
    priority: 3
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)
    assert cfg.models["thinker"].model_class == "reasoning"
    assert cfg.models["chatter"].model_class == "chat"
    # Default when 'class' key is absent
    assert cfg.models["default-class"].model_class == "chat"


def test_model_class_invalid_falls_back_to_chat(tmp_path):
    yaml_text = """
models:
  weird:
    tags: [code]
    class: superintelligent
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)
    # Unknown class values fall back to "chat" rather than raising.
    assert cfg.models["weird"].model_class == "chat"


def test_malformed_yaml_fallback(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ollama: [not_a_dict\n")
    cfg = load_config(config_path)
    assert isinstance(cfg, Config)
    assert cfg.ollama.base_url == "http://localhost:11434"
    assert cfg.models == {}
    assert cfg.thresholds.single_confidence == 0.8
    assert cfg.classifier.embeddings_model == "all-MiniLM-L6-v2"

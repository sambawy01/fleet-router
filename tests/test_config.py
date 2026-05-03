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
    assert cfg.thresholds.single_confidence == 0.8
    assert cfg.classifier.embeddings_model == "all-MiniLM-L6-v2"

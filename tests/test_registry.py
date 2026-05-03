import pytest
from fleet.registry import ModelRegistry
from fleet.config import Config, ModelEntry

def test_get_best_for_tag():
    config = Config(models={
        "deepseek-v4-pro": ModelEntry(tags=["code", "reasoning"], priority=1),
        "glm-5.1": ModelEntry(tags=["creative"], priority=2),
    })
    reg = ModelRegistry(config)
    # Mock _fetch_available to avoid network call
    reg._available = {"deepseek-v4-pro", "glm-5.1"}

    assert reg.get_best_for_tag("code") == "deepseek-v4-pro"
    assert reg.get_best_for_tag("creative") == "glm-5.1"
    assert reg.get_best_for_tag("unknown") is None

def test_models_for_parallel():
    config = Config(models={
        "deepseek-v4-pro": ModelEntry(tags=["code", "reasoning"], priority=1),
        "glm-5.1": ModelEntry(tags=["creative", "code"], priority=2),
        "minimax-m2.7": ModelEntry(tags=["summarize"], priority=3),
    })
    reg = ModelRegistry(config)
    reg._available = {"deepseek-v4-pro", "glm-5.1", "minimax-m2.7"}

    models = reg.models_for_tag("code", top_n=2)
    assert models == ["deepseek-v4-pro", "glm-5.1"]

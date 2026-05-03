import pytest
import requests
from unittest.mock import patch, MagicMock
from fleet.registry import ModelRegistry
from fleet.config import Config, ModelEntry, OllamaConfig


def test_get_best_for_tag():
    config = Config(models={
        "deepseek-v4-pro": ModelEntry(tags=["code", "reasoning"], priority=1),
        "glm-5.1": ModelEntry(tags=["creative"], priority=2),
    })
    reg = ModelRegistry(config)
    # Set available models directly
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


def test_models_for_tag_sort_order():
    config = Config(models={
        "alpha": ModelEntry(tags=["general"], priority=3),
        "beta": ModelEntry(tags=["general"], priority=1),
        "gamma": ModelEntry(tags=["general"], priority=2),
    })
    reg = ModelRegistry(config)
    reg._available = {"alpha", "beta", "gamma"}

    models = reg.models_for_tag("general", top_n=3)
    # Verify ascending priority order
    assert models == ["beta", "gamma", "alpha"]


def test_models_for_tag_top_n_larger_than_matches():
    config = Config(models={
        "model-a": ModelEntry(tags=["only"], priority=1),
    })
    reg = ModelRegistry(config)
    reg._available = {"model-a"}

    models = reg.models_for_tag("only", top_n=10)
    assert models == ["model-a"]


def test_models_for_tag_no_match():
    config = Config(models={
        "model-a": ModelEntry(tags=["code"], priority=1),
    })
    reg = ModelRegistry(config)
    reg._available = {"model-a"}

    models = reg.models_for_tag("notag", top_n=3)
    assert models == []


def test_all_available():
    config = Config(models={
        "deepseek-v4-pro": ModelEntry(tags=["code"], priority=1),
        "glm-5.1": ModelEntry(tags=["creative"], priority=2),
    })
    reg = ModelRegistry(config)
    reg._available = {"glm-5.1", "deepseek-v4-pro"}

    assert reg.all_available() == ["deepseek-v4-pro", "glm-5.1"]


@patch("fleet.registry.requests.get")
def test_refresh_failure_yields_empty_set(mock_get):
    """On network failure the registry must NOT pretend all configured models
    are installed — that produces guaranteed downstream 404s."""
    config = Config(
        ollama=OllamaConfig(base_url="http://localhost:11434"),
        models={
            "deepseek-v4-pro": ModelEntry(tags=["code"], priority=1),
            "glm-5.1": ModelEntry(tags=["creative"], priority=2),
        },
    )
    mock_get.side_effect = requests.RequestException("Connection refused")

    reg = ModelRegistry(config)
    reg.refresh()

    assert reg._available == set()


@patch("fleet.registry.requests.get")
def test_refresh_skips_entries_without_name(mock_get):
    """Ollama responses missing the 'name' key must not raise KeyError."""
    config = Config(
        ollama=OllamaConfig(base_url="http://localhost:11434"),
        models={"glm-5.1": ModelEntry(tags=["creative"], priority=2)},
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {"name": "glm-5.1:fp16"},
            {},
            {"name": ""},
            {"name": None},
            "not-a-dict",
        ]
    }
    mock_get.return_value = mock_response

    reg = ModelRegistry(config)
    reg.refresh()

    assert reg._available == {"glm-5.1"}


@patch("fleet.registry.requests.get")
def test_refresh_success(mock_get):
    config = Config(
        ollama=OllamaConfig(base_url="http://localhost:11434"),
        models={
            "deepseek-v4-pro": ModelEntry(tags=["code"], priority=1),
            "glm-5.1": ModelEntry(tags=["creative"], priority=2),
        },
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {"name": "deepseek-v4-pro:latest"},
            {"name": "glm-5.1:fp16"},
        ]
    }
    mock_get.return_value = mock_response

    reg = ModelRegistry(config)
    reg.refresh()

    # Verify model names have tag suffixes stripped
    assert reg._available == {"deepseek-v4-pro", "glm-5.1"}


@patch("fleet.registry.requests.get")
def test_refresh_passes_api_key(mock_get):
    """When ollama.api_key is set, registry refresh must include Authorization header."""
    config = Config(
        ollama=OllamaConfig(base_url="http://localhost:11434", api_key="sk-secret"),
        models={"glm-5.1": ModelEntry(tags=["creative"], priority=2)},
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "glm-5.1:fp16"}]}
    mock_get.return_value = mock_response

    reg = ModelRegistry(config)
    reg.refresh()

    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs.get("headers") == {"Authorization": "Bearer sk-secret"}
    assert reg._available == {"glm-5.1"}


@patch("fleet.registry.requests.get")
def test_refresh_no_api_key_no_auth_header(mock_get):
    """When ollama.api_key is empty, no Authorization header should be sent."""
    config = Config(
        ollama=OllamaConfig(base_url="http://localhost:11434", api_key=""),
        models={"glm-5.1": ModelEntry(tags=["creative"], priority=2)},
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "glm-5.1:fp16"}]}
    mock_get.return_value = mock_response

    reg = ModelRegistry(config)
    reg.refresh()

    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs.get("headers") == {}
    assert reg._available == {"glm-5.1"}

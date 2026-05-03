import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from fleet.router import (
    FleetRouter,
    ERROR_MODEL_FAILED,
    ERROR_ALL_MODELS_FAILED,
    ERROR_NO_MODEL,
    ERROR_NO_MODELS,
)
from fleet.config import Config


@pytest.fixture
def router():
    config = Config()
    return FleetRouter(config)


@pytest.mark.asyncio
async def test_single_mode(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="deepseek-v4-pro"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"deepseek-v4-pro": "def foo():\n    pass"}

        result = await router.ask("write a function")
        assert result == "def foo():\n    pass"
        mock_dispatch.assert_awaited_once()
        # Should only call 1 model
        assert len(mock_dispatch.call_args[0][1]) == 1


@pytest.mark.asyncio
async def test_parallel_mode(router):
    with patch.object(router._classifier, "classify", return_value=("creative", 0.6)), \
         patch.object(router._registry, "models_for_tag", return_value=["glm-5.1", "minimax-m2.7"]), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch, \
         patch.object(router._synthesizer, "pick", return_value="best result"):
        mock_dispatch.return_value = {"glm-5.1": "a", "minimax-m2.7": "b"}

        result = await router.ask("write a story")
        assert result == "best result"
        # Should call 2 models
        assert len(mock_dispatch.call_args[0][1]) == 2


@pytest.mark.asyncio
async def test_force_model(router):
    with patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"gpt-4": "forced result"}

        result = await router.ask("prompt", force_model="gpt-4")
        assert result == "forced result"
        mock_dispatch.assert_awaited_once()
        assert mock_dispatch.call_args[0][1] == ["gpt-4"]


@pytest.mark.asyncio
async def test_force_parallel(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "models_for_tag", return_value=["model-a", "model-b"]), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch, \
         patch.object(router._synthesizer, "pick", return_value="parallel result"):
        mock_dispatch.return_value = {"model-a": "a", "model-b": "b"}

        result = await router.ask("prompt", force_parallel=True)
        assert result == "parallel result"
        assert len(mock_dispatch.call_args[0][1]) == 2


@pytest.mark.asyncio
async def test_single_fallback_when_primary_fails(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="primary-model"), \
         patch.object(router._registry, "all_available", return_value=["fallback-model"]), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.side_effect = [
            {"primary-model": None},
            {"fallback-model": "fallback result"},
        ]

        result = await router.ask("prompt")
        assert result == "fallback result"
        assert mock_dispatch.await_count == 2


@pytest.mark.asyncio
async def test_no_models_available(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.6)), \
         patch.object(router._registry, "models_for_tag", return_value=[]), \
         patch.object(router._registry, "all_available", return_value=[]):
        result = await router.ask("prompt")
        assert result == ERROR_NO_MODELS


@pytest.mark.asyncio
async def test_system_prompt_forwarded(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="model"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"model": "result"}

        result = await router.ask("prompt", system="You are helpful")
        assert result == "result"
        assert mock_dispatch.call_args.kwargs["system"] == "You are helpful"


@pytest.mark.asyncio
async def test_empty_string_response_not_treated_as_failure(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="model"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"model": ""}

        result = await router.ask("prompt")
        assert result == ""

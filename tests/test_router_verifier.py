"""Verifier-mode behavior of FleetRouter (the new default).

Existing heuristic-mode behavior lives in test_router.py — these tests cover
the verifier path: self-consistency dispatch, abstention, escalation, and
refinement.
"""
from unittest.mock import AsyncMock, patch

import pytest

from fleet.config import (
    Config,
    EscalationConfig,
    ModelEntry,
    RefinementConfig,
    SamplingConfig,
    SynthesisConfig,
)
from fleet.router import FleetRouter
from fleet.verifiers.base import Candidate, VerificationResult


@pytest.fixture
def config():
    return Config(
        models={
            "model-a": ModelEntry(tags=["math"], priority=1),
            "model-b": ModelEntry(tags=["math"], priority=2),
        },
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"math": 3, "default": 1}),
    )


@pytest.fixture
def router(config):
    r = FleetRouter(config)
    r._registry._available = {"model-a", "model-b"}
    r._registry._refreshed = True
    return r


@pytest.mark.asyncio
async def test_verifier_path_uses_run_multi_with_configured_samples(router):
    """sampling.samples_by_tag['math'] = 3 → run_multi(samples=3)."""
    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["the answer is 7"], "model-b": ["the answer is 7"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-a", 0, "the answer is 7", score=0.9),
            all_scored=[],
        )
        result = await router.ask("solve 5+2")
    assert result == "the answer is 7"
    assert mock_multi.call_args.kwargs["samples"] == 3


@pytest.mark.asyncio
async def test_verifier_abstention_returns_calibrated_uncertainty(router):
    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["1"], "model-b": ["2"]}
        mock_pick.return_value = VerificationResult(
            winner=None,
            all_scored=[
                Candidate("model-a", 0, "answer 1", score=0.3),
                Candidate("model-b", 0, "answer 2", score=0.3),
            ],
            rationale="no majority",
            abstain=True,
        )
        result = await router.ask("solve 5+2")
    assert "uncertain" in result
    assert "no majority" in result
    assert "model-a" in result and "model-b" in result


@pytest.mark.asyncio
async def test_escalation_runs_when_verifier_abstains():
    config = Config(
        models={
            "model-a": ModelEntry(tags=["math"], priority=1),
            "model-b": ModelEntry(tags=["math"], priority=2),
            "judge": ModelEntry(tags=["math"], priority=3),
        },
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"math": 1, "default": 1}),
        escalation=EscalationConfig(enabled=True, model="judge", score_threshold=0.6),
    )
    router = FleetRouter(config)
    router._registry._available = {"model-a", "model-b", "judge"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["1"], "model-b": ["2"]}
        mock_pick.return_value = VerificationResult(
            winner=None,
            all_scored=[Candidate("model-a", 0, "1", score=0.2)],
            abstain=True,
        )
        mock_run.return_value = {"judge": "the correct synthesized answer"}
        result = await router.ask("solve")
    assert result == "the correct synthesized answer"
    # Escalation called dispatcher.run with the judge model.
    assert mock_run.call_args[0][1] == ["judge"]


@pytest.mark.asyncio
async def test_refinement_runs_critique_then_revise():
    config = Config(
        models={"model-a": ModelEntry(tags=["general"], priority=1),
                "critic": ModelEntry(tags=["general"], priority=2)},
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"default": 1}),
        refinement=RefinementConfig(enabled=True, critique_model="critic"),
    )
    router = FleetRouter(config)
    router._registry._available = {"model-a", "critic"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("general", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["draft answer"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-a", 0, "draft answer", score=0.8),
            all_scored=[],
        )
        mock_run.side_effect = [
            {"critic": "you forgot to mention X"},  # critique
            {"critic": "draft answer plus X"},      # revise
        ]
        result = await router.ask("explain something")
    assert result == "draft answer plus X"
    assert mock_run.await_count == 2


@pytest.mark.asyncio
async def test_refinement_skipped_on_no_critique_needed():
    config = Config(
        models={"model-a": ModelEntry(tags=["general"], priority=1),
                "critic": ModelEntry(tags=["general"], priority=2)},
        synthesis=SynthesisConfig(mode="verifier"),
        refinement=RefinementConfig(enabled=True, critique_model="critic"),
    )
    router = FleetRouter(config)
    router._registry._available = {"model-a", "critic"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("general", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["good draft"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-a", 0, "good draft", score=0.9),
            all_scored=[],
        )
        mock_run.return_value = {"critic": "no critique needed"}
        result = await router.ask("explain")
    # Only the critique call was made; no revise call because critic said it was fine.
    assert result == "good draft"
    assert mock_run.await_count == 1

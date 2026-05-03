from unittest.mock import AsyncMock

import pytest

from fleet.verifiers.base import Candidate
from fleet.verifiers.judge import JudgeVerifier, _extract_json


def test_extract_json_pure_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_embedded_in_prose():
    text = "Here is the result:\n\n{\"best\": \"A\", \"scores\": {\"A\": 8}}\n\nThanks!"
    assert _extract_json(text) == {"best": "A", "scores": {"A": 8}}


def test_extract_json_returns_none_when_no_json():
    assert _extract_json("just text") is None


def test_extract_json_handles_nested():
    text = '{"outer": {"inner": [1, 2, 3]}}'
    assert _extract_json(text) == {"outer": {"inner": [1, 2, 3]}}


@pytest.mark.asyncio
async def test_judge_picks_best_and_normalizes_scores():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 8, "B": 3}, "best": "A", "rationale": "A is better"}'
    ])
    v = JudgeVerifier(provider=provider, judge_model="judge", tag="general")
    candidates = [
        Candidate("model-a", 0, "answer A"),
        Candidate("model-b", 0, "answer B"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.model == "model-a"
    assert result.winner.score == 0.8
    assert "A is better" in result.rationale


@pytest.mark.asyncio
async def test_judge_falls_back_to_first_candidate_on_unparseable_output():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=["lol the model just emitted prose"])
    v = JudgeVerifier(provider=provider, judge_model="judge")
    candidates = [
        Candidate("a", 0, "first"),
        Candidate("b", 0, "second"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.model == "a"
    assert "unparseable" in result.rationale


@pytest.mark.asyncio
async def test_judge_falls_back_when_provider_returns_nothing():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=[None])
    v = JudgeVerifier(provider=provider, judge_model="judge")
    candidates = [Candidate("a", 0, "x"), Candidate("b", 0, "y")]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.model == "a"


@pytest.mark.asyncio
async def test_judge_handles_unknown_best_label():
    """When the judge points at a label that doesn't exist, fall back to highest-scored."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 4, "B": 9}, "best": "Z"}'
    ])
    v = JudgeVerifier(provider=provider, judge_model="judge")
    candidates = [Candidate("a", 0, "x"), Candidate("b", 0, "y")]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.model == "b"  # highest score


@pytest.mark.asyncio
async def test_judge_passes_single_candidate_through():
    provider = AsyncMock()
    v = JudgeVerifier(provider=provider, judge_model="judge")
    result = await v.aggregate("p", [Candidate("a", 0, "only")])
    assert result.winner is not None
    assert result.winner.model == "a"
    # No judge call needed for a single candidate.
    provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_judge_handles_provider_exception():
    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=RuntimeError("boom"))
    v = JudgeVerifier(provider=provider, judge_model="judge")
    candidates = [Candidate("a", 0, "x"), Candidate("b", 0, "y")]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None  # graceful fallback

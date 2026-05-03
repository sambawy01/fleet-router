import pytest

from fleet.verifiers.base import Candidate
from fleet.verifiers.heuristic import HeuristicVerifier


@pytest.mark.asyncio
async def test_heuristic_picks_via_underlying_synthesizer():
    v = HeuristicVerifier(tag="code")
    candidates = [
        Candidate("a", 0, "broken syntax ("),
        Candidate("b", 0, "def foo():\n    return 1"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert "def foo" in result.winner.text


@pytest.mark.asyncio
async def test_heuristic_abstains_on_synthesizer_tie():
    """When the heuristic synthesizer returns a dict (tie), wrap as abstention."""
    v = HeuristicVerifier(tag="general")
    candidates = [
        Candidate("a", 0, "abc"),  # length 3
        Candidate("b", 0, "def"),  # length 3 — tie
    ]
    result = await v.aggregate("p", candidates)
    assert result.abstain


@pytest.mark.asyncio
async def test_heuristic_handles_empty_candidates():
    v = HeuristicVerifier()
    result = await v.aggregate("p", [])
    assert result.abstain

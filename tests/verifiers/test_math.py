import pytest

from fleet.verifiers.base import Candidate
from fleet.verifiers.math import MathVerifier, _extract_final_answer


def test_extract_final_answer_explicit_marker():
    assert _extract_final_answer("So the answer is 42.") == "42"
    assert _extract_final_answer("Final answer: 3.14") == "3.14"
    assert _extract_final_answer(r"\boxed{17}") == "17"


def test_extract_final_answer_falls_back_to_last_number():
    text = "First we compute 10, then 20, then 30. So 30."
    assert _extract_final_answer(text) == "30"


def test_extract_final_answer_handles_decimals_and_negatives():
    assert _extract_final_answer("Answer: -2.5") == "-2.5"
    assert _extract_final_answer("Answer: 1.5e3") == "1500"


def test_extract_final_answer_normalizes():
    """42 and 42.0 should be considered the same numeric answer."""
    assert _extract_final_answer("Answer: 42.0") == "42"


def test_extract_final_answer_returns_none_on_no_number():
    assert _extract_final_answer("I don't know.") is None


@pytest.mark.asyncio
async def test_math_verifier_majority_vote_picks_winner():
    v = MathVerifier()
    candidates = [
        Candidate("a", 0, "Answer: 42"),
        Candidate("b", 0, "I think the answer is 42."),
        Candidate("c", 0, "The answer is 99."),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert "42" in result.winner.text
    assert not result.abstain


@pytest.mark.asyncio
async def test_math_verifier_abstains_on_no_majority():
    """3 candidates, 3 different answers → no majority → abstain."""
    v = MathVerifier()
    candidates = [
        Candidate("a", 0, "Answer: 1"),
        Candidate("b", 0, "Answer: 2"),
        Candidate("c", 0, "Answer: 3"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.abstain


@pytest.mark.asyncio
async def test_math_verifier_abstains_on_no_extractable_numbers():
    v = MathVerifier()
    candidates = [
        Candidate("a", 0, "I don't know"),
        Candidate("b", 0, "no idea"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.abstain


@pytest.mark.asyncio
async def test_math_verifier_strips_thinking_before_extract():
    v = MathVerifier()
    candidates = [
        Candidate("a", 0, "<think>let me work this out: 1+1=2, no wait, 2+2=4</think>The answer is 7."),
        Candidate("b", 0, "Answer: 7"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    # Both should agree on 7 — thinking tokens shouldn't pollute extraction.
    assert "7" in result.winner.text

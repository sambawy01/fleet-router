import pytest

from evals.scorers.base import EvalCase
from evals.scorers.multi_choice import MultipleChoiceScorer, _extract_letter


def test_extract_letter_explicit_marker():
    assert _extract_letter("The answer is B.") == "B"
    assert _extract_letter("Answer: C") == "C"
    assert _extract_letter("Final answer: A") == "A"


def test_extract_letter_parenthesized():
    assert _extract_letter("After analysis, (D) is correct.") == "D"


def test_extract_letter_letter_alone():
    """When the model answers with just the letter on a line."""
    assert _extract_letter("Some reasoning here.\n\nB") == "B"
    assert _extract_letter("B.") == "B"


def test_extract_letter_strips_thinking():
    text = "<think>let me work through this... A seems wrong, then B...</think>\n\nThe answer is C."
    assert _extract_letter(text) == "C"


def test_extract_letter_handles_no_match():
    assert _extract_letter("I don't know.") is None
    assert _extract_letter("") is None


def test_extract_letter_case_insensitive():
    assert _extract_letter("the answer is b") == "B"


@pytest.mark.asyncio
async def test_score_correct():
    s = MultipleChoiceScorer()
    case = EvalCase(prompt="q?", tag="reasoning", expected="B", scorer="multi_choice")
    result = await s.score(case, "After thinking, the answer is B.")
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_score_incorrect():
    s = MultipleChoiceScorer()
    case = EvalCase(prompt="q?", tag="reasoning", expected="B", scorer="multi_choice")
    result = await s.score(case, "The answer is A.")
    assert result.score == 0.0
    assert "got=A, want=B" in result.notes


@pytest.mark.asyncio
async def test_score_no_letter_extracted():
    s = MultipleChoiceScorer()
    case = EvalCase(prompt="q?", tag="reasoning", expected="B", scorer="multi_choice")
    result = await s.score(case, "I'm not sure.")
    assert result.score == 0.0
    assert "no letter found" in result.notes

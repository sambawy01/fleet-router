"""Tests that EvalCase.scorer overrides the tag default scorer."""
from pathlib import Path

import pytest

from evals.runner import default_scorers, load_fixtures, run_eval
from evals.scorers import EvalCase


class _StubRouter:
    def __init__(self, answer): self._a = answer
    async def ask(self, prompt): return self._a


def test_load_fixtures_reads_scorer_field(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text(
        '{"prompt": "q1", "tag": "reasoning", "expected": "B", "scorer": "multi_choice"}\n'
        '{"prompt": "q2", "tag": "reasoning", "expected": ["foo"]}\n'
    )
    cases = load_fixtures(tmp_path)
    assert cases[0].scorer == "multi_choice"
    assert cases[1].scorer == ""


@pytest.mark.asyncio
async def test_runner_uses_explicit_scorer_when_set():
    """A reasoning-tagged case with scorer='multi_choice' should be scored as
    multi-choice, not keyword-contains."""
    cases = [
        EvalCase(
            prompt="q1", tag="reasoning",
            expected="C", scorer="multi_choice",
        ),
    ]
    router = _StubRouter("After deliberation, the answer is C.")
    results = await run_eval(router, cases, default_scorers())
    assert results[0].score == 1.0  # multi_choice scorer matched
    assert "got=C, want=C" in results[0].notes


@pytest.mark.asyncio
async def test_runner_falls_back_to_tag_scorer_when_no_override():
    cases = [
        EvalCase(prompt="q1", tag="reasoning", expected=["python", "function"]),
    ]
    router = _StubRouter("python function in 3 lines")
    results = await run_eval(router, cases, default_scorers())
    assert results[0].score == 1.0  # keyword-contains hit both

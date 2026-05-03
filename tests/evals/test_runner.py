"""Tests for the eval harness — uses a stub router so no Ollama needed."""
from pathlib import Path

import pytest

from evals.runner import (
    aggregate,
    compare_to_baseline,
    load_fixtures,
    run_eval,
    save_baseline,
)
from evals.scorers import EvalCase, EvalResult, KeywordContainsScorer, NumericMatchScorer


class _StubRouter:
    """Returns a fixed answer per prompt prefix."""

    def __init__(self, answers: dict[str, str]):
        self._answers = answers

    async def ask(self, prompt: str):
        for prefix, ans in self._answers.items():
            if prompt.startswith(prefix):
                return ans
        return "no answer"


def test_load_fixtures_reads_jsonl(tmp_path):
    f = tmp_path / "math.jsonl"
    f.write_text(
        '{"prompt": "1+1?", "tag": "math", "expected": 2}\n'
        '{"prompt": "2*3?", "tag": "math", "expected": 6}\n'
    )
    cases = load_fixtures(tmp_path)
    assert len(cases) == 2
    assert cases[0].prompt == "1+1?"
    assert cases[0].expected == 2


def test_load_fixtures_skips_blank_and_comments(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text(
        '\n'
        '# this is a comment\n'
        '{"prompt": "p", "tag": "general", "expected": ["x"]}\n'
    )
    assert len(load_fixtures(tmp_path)) == 1


def test_load_fixtures_raises_on_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fixtures(tmp_path / "nope")


@pytest.mark.asyncio
async def test_run_eval_scores_each_case():
    cases = [
        EvalCase(prompt="math1", tag="math", expected=42),
        EvalCase(prompt="math2", tag="math", expected=10),
    ]
    router = _StubRouter({"math1": "the answer is 42", "math2": "I think 99"})
    scorers = {"math": NumericMatchScorer()}
    results = await run_eval(router, cases, scorers)
    assert len(results) == 2
    assert results[0].score == 1.0
    assert results[1].score == 0.0


@pytest.mark.asyncio
async def test_run_eval_handles_dict_answer():
    cases = [EvalCase(prompt="q", tag="general", expected=["foo"])]
    class DictRouter:
        async def ask(self, p):
            return {"a": "foo content", "b": "other"}
    results = await run_eval(DictRouter(), cases, {"general": KeywordContainsScorer()})
    assert results[0].score == 1.0


def test_aggregate_per_tag():
    results = [
        EvalResult(case=EvalCase(prompt="", tag="math"), answer="", score=1.0),
        EvalResult(case=EvalCase(prompt="", tag="math"), answer="", score=0.0),
        EvalResult(case=EvalCase(prompt="", tag="code"), answer="", score=1.0),
    ]
    agg = aggregate(results)
    assert agg["math"]["n"] == 2
    assert agg["math"]["mean_score"] == 0.5
    assert agg["math"]["pass_rate"] == 0.5
    assert agg["code"]["pass_rate"] == 1.0


def test_compare_to_baseline_no_baseline(tmp_path):
    regressed, msgs = compare_to_baseline({}, tmp_path / "missing.json")
    assert not regressed
    assert "no baseline" in msgs[0]


def test_compare_to_baseline_detects_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    save_baseline({"math": {"n": 5, "mean_score": 0.9, "pass_rate": 0.9}}, baseline)
    current = {"math": {"n": 5, "mean_score": 0.5, "pass_rate": 0.5}}
    regressed, msgs = compare_to_baseline(current, baseline, regression_pp=3.0)
    assert regressed
    assert any("math" in m for m in msgs)


def test_compare_to_baseline_within_tolerance(tmp_path):
    baseline = tmp_path / "b.json"
    save_baseline({"math": {"n": 5, "mean_score": 0.8, "pass_rate": 0.8}}, baseline)
    current = {"math": {"n": 5, "mean_score": 0.78, "pass_rate": 0.79}}
    regressed, _ = compare_to_baseline(current, baseline, regression_pp=3.0)
    assert not regressed

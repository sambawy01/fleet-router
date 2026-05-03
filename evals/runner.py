"""Eval harness — runs fixture cases through a router and scores them.

Two output forms:
- Per-case results (for debugging individual failures)
- Per-tag aggregates (mean score, count, runtime) compared against baseline
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from evals.scorers import (
    CodeExecScorer,
    EvalCase,
    EvalResult,
    KeywordContainsScorer,
    MultipleChoiceScorer,
    NumericMatchScorer,
    Scorer,
)

logger = logging.getLogger(__name__)


def default_scorers() -> dict[str, Scorer]:
    """Tag-default and explicit-name scorers. EvalCase.scorer overrides the
    tag default — useful when one tag (e.g. reasoning) needs multiple
    scoring methods across cases."""
    return {
        # Tag defaults
        "code": CodeExecScorer(),
        "math": NumericMatchScorer(),
        "reasoning": KeywordContainsScorer(),
        "summarize": KeywordContainsScorer(),
        "creative": KeywordContainsScorer(),
        "translate": KeywordContainsScorer(),
        "general": KeywordContainsScorer(),
        # Explicit-override names (set EvalCase.scorer to one of these)
        "multi_choice": MultipleChoiceScorer(),
        "code_exec": CodeExecScorer(),
        "numeric": NumericMatchScorer(),
        "keyword": KeywordContainsScorer(),
    }


def load_fixtures(directory: Path | str) -> list[EvalCase]:
    """Load all *.jsonl files under `directory`. Each line = one case."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"fixtures directory not found: {directory}")
    cases: list[EvalCase] = []
    for path in sorted(directory.glob("*.jsonl")):
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s line %d: %s — skipping", path, i, exc)
                    continue
                cases.append(EvalCase(
                    prompt=str(raw.get("prompt", "")),
                    tag=str(raw.get("tag", "general")),
                    expected=raw.get("expected"),
                    test_code=str(raw.get("test_code", "")),
                    scorer=str(raw.get("scorer", "")),
                    metadata=raw.get("metadata", {}) or {},
                ))
    return cases


async def _answer_to_str(router, prompt: str) -> str:
    answer = await router.ask(prompt)
    if isinstance(answer, dict):
        return "\n\n".join(f"--- {k} ---\n{v}" for k, v in answer.items())
    return str(answer)


async def run_eval(
    router,
    cases: list[EvalCase],
    scorers: Optional[dict[str, Scorer]] = None,
) -> list[EvalResult]:
    """Run every case through `router`, score with the per-tag scorer.
    Returns results in input order."""
    scorers = scorers or default_scorers()
    results: list[EvalResult] = []
    for case in cases:
        scorer = scorers.get(case.scorer) if case.scorer else None
        if scorer is None:
            scorer = scorers.get(case.tag)
        if scorer is None:
            logger.warning("no scorer for tag=%s scorer=%s; skipping case",
                           case.tag, case.scorer)
            continue
        answer = await _answer_to_str(router, case.prompt)
        result = await scorer.score(case, answer)
        results.append(result)
    return results


def aggregate(results: list[EvalResult]) -> dict[str, dict]:
    """Aggregate by tag: mean score, count, pass rate."""
    by_tag: dict[str, list[EvalResult]] = {}
    for r in results:
        by_tag.setdefault(r.case.tag, []).append(r)
    out: dict[str, dict] = {}
    for tag, rs in by_tag.items():
        scores = [r.score for r in rs]
        passes = sum(1 for s in scores if s >= 0.5)
        out[tag] = {
            "n": len(rs),
            "mean_score": sum(scores) / len(scores),
            "pass_rate": passes / len(rs),
        }
    return out


def compare_to_baseline(
    current: dict[str, dict],
    baseline_path: Path | str,
    regression_pp: float = 3.0,
) -> tuple[bool, list[str]]:
    """Return (regressed, messages). `regression_pp` is percentage points of
    pass-rate that count as a regression."""
    path = Path(baseline_path)
    if not path.exists():
        return False, [f"no baseline at {path}; current saved as new baseline"]
    with open(path) as f:
        baseline = json.load(f)
    regressed = False
    messages: list[str] = []
    for tag, agg in current.items():
        b = baseline.get(tag)
        if b is None:
            messages.append(f"{tag}: new tag (no baseline)")
            continue
        delta_pp = (agg["pass_rate"] - b.get("pass_rate", 0.0)) * 100
        sign = "+" if delta_pp >= 0 else ""
        messages.append(
            f"{tag}: pass_rate {agg['pass_rate']:.0%} ({sign}{delta_pp:.1f}pp) "
            f"mean {agg['mean_score']:.2f} (n={agg['n']})"
        )
        if delta_pp < -regression_pp:
            regressed = True
    return regressed, messages


def save_baseline(aggregates: dict[str, dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(aggregates, f, indent=2)


async def run_and_report(
    router,
    fixtures_dir: Path | str,
    baseline_path: Optional[Path | str] = None,
) -> dict:
    """End-to-end: load → run → aggregate → compare. Returns a report dict."""
    cases = load_fixtures(fixtures_dir)
    start = time.time()
    results = await run_eval(router, cases)
    elapsed = time.time() - start
    aggregates = aggregate(results)
    report: dict = {
        "n_cases": len(cases),
        "elapsed_s": elapsed,
        "aggregates": aggregates,
        "results": [
            {
                "tag": r.case.tag,
                "prompt": r.case.prompt[:200],
                "score": r.score,
                "notes": r.notes,
            }
            for r in results
        ],
    }
    if baseline_path is not None:
        regressed, messages = compare_to_baseline(aggregates, baseline_path)
        report["regressed"] = regressed
        report["comparison"] = messages
    return report

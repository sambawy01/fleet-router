"""Side-by-side comparison harness.

Runs the same fixture set through two routers and reports per-tag deltas.
The "router" is anything with an `async def ask(prompt) -> str` method —
swap in fleet for one side, and any other LLM caller (Anthropic SDK, OpenAI
SDK, or another fleet config) for the other.

Usage:

    import asyncio
    from fleet import FleetRouter, load_config
    from evals.compare import compare

    fleet_router = FleetRouter(load_config())

    class OpusRouter:
        def __init__(self, client): self._c = client
        async def ask(self, prompt):
            msg = self._c.messages.create(
                model="claude-opus-4-7",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

    # Provide your own Anthropic client; fleet does not ship one.
    opus_router = OpusRouter(my_anthropic_client)

    report = asyncio.run(compare(
        a=("fleet", fleet_router),
        b=("opus", opus_router),
        fixtures_dir="evals/fixtures/",
    ))
    print(report["summary"])
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from evals.runner import aggregate, default_scorers, load_fixtures
from evals.scorers import EvalCase, EvalResult, Scorer


async def _run_one_side(
    label: str,
    router,
    cases: list[EvalCase],
    scorers: dict[str, Scorer],
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        scorer = scorers.get(case.tag)
        if scorer is None:
            continue
        try:
            answer = await router.ask(case.prompt)
            if isinstance(answer, dict):
                answer = "\n\n".join(f"--- {k} ---\n{v}" for k, v in answer.items())
            answer = str(answer)
        except Exception as exc:  # noqa: BLE001
            answer = f"(error: {type(exc).__name__}: {exc})"
        result = await scorer.score(case, answer)
        results.append(result)
    return results


async def compare(
    a: tuple[str, object],
    b: tuple[str, object],
    fixtures_dir: Path | str,
    scorers: Optional[dict[str, Scorer]] = None,
) -> dict:
    """Run two routers through the same fixtures; report per-tag deltas."""
    label_a, router_a = a
    label_b, router_b = b
    cases = load_fixtures(fixtures_dir)
    scorers = scorers or default_scorers()

    # Run both sides concurrently — saves wall time when networks dominate.
    results_a, results_b = await asyncio.gather(
        _run_one_side(label_a, router_a, cases, scorers),
        _run_one_side(label_b, router_b, cases, scorers),
    )

    agg_a = aggregate(results_a)
    agg_b = aggregate(results_b)

    # Per-case head-to-head
    head_to_head: list[dict] = []
    for ra, rb in zip(results_a, results_b):
        head_to_head.append({
            "tag": ra.case.tag,
            "prompt": ra.case.prompt[:120],
            f"{label_a}_score": ra.score,
            f"{label_b}_score": rb.score,
            "delta": rb.score - ra.score,
        })

    # Per-tag delta + win counts
    summary_lines: list[str] = []
    summary_lines.append(f"Comparison: {label_a} vs {label_b}\n")
    all_tags = sorted(set(agg_a) | set(agg_b))
    for tag in all_tags:
        a_pass = agg_a.get(tag, {}).get("pass_rate", 0.0)
        b_pass = agg_b.get(tag, {}).get("pass_rate", 0.0)
        delta_pp = (b_pass - a_pass) * 100
        winner = label_b if delta_pp > 0 else (label_a if delta_pp < 0 else "tie")
        summary_lines.append(
            f"  {tag:12s}  {label_a}={a_pass:.0%}  {label_b}={b_pass:.0%}  "
            f"delta={delta_pp:+.1f}pp  → {winner}"
        )

    a_wins = sum(1 for r in head_to_head if r["delta"] < 0)
    b_wins = sum(1 for r in head_to_head if r["delta"] > 0)
    ties = sum(1 for r in head_to_head if r["delta"] == 0)
    summary_lines.append(
        f"\nHead-to-head: {label_a}={a_wins}  {label_b}={b_wins}  ties={ties}"
    )

    return {
        "labels": [label_a, label_b],
        "n_cases": len(cases),
        "aggregates": {label_a: agg_a, label_b: agg_b},
        "head_to_head": head_to_head,
        "summary": "\n".join(summary_lines),
    }

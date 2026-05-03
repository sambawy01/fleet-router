"""CLI entrypoint for direct testing outside Claude Code.

Subcommands:
- (default) ask        — route a single prompt
- eval                 — run the eval harness against the fixtures directory
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from fleet.config import clean_model_key, load_config
from fleet.router import FleetRouter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fleet", description="Fleet Router CLI")
    parser.add_argument("prompt", nargs="*", help="Prompt text (omitted only with --eval)")
    parser.add_argument("--parallel", action="store_true", help="Force parallel mode")
    parser.add_argument("--model", default=None, help="Force specific model")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging to stderr",
    )
    parser.add_argument(
        "--eval", default=None, metavar="FIXTURES_DIR",
        help="Run eval harness on the given fixtures directory",
    )
    parser.add_argument(
        "--baseline", default=None, metavar="PATH",
        help="Eval mode only: path to baseline JSON for regression comparison",
    )
    parser.add_argument(
        "--save-baseline", default=None, metavar="PATH",
        help="Eval mode only: save current aggregates as the new baseline",
    )
    return parser


def _run_ask(router: FleetRouter, args: argparse.Namespace) -> int:
    if not args.prompt:
        print("fleet: missing prompt (or use --eval)", file=sys.stderr)
        return 1
    prompt = " ".join(args.prompt)
    force_model = clean_model_key(args.model) if args.model else None
    try:
        result = asyncio.run(router.ask(
            prompt,
            force_parallel=args.parallel,
            force_model=force_model,
        ))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 — top-level CLI guard
        print(f"fleet: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if isinstance(result, dict):
        for model, text in result.items():
            print(f"\n--- {model} ---\n{text}\n")
    else:
        print(result)

    if isinstance(result, str) and result.startswith("("):
        return 2
    return 0


def _run_eval(router: FleetRouter, args: argparse.Namespace) -> int:
    # Lazy import — eval module not needed for the common ask path.
    from evals.runner import (
        aggregate, compare_to_baseline, load_fixtures, run_eval, save_baseline,
    )

    fixtures_dir = Path(args.eval)
    try:
        cases = load_fixtures(fixtures_dir)
    except FileNotFoundError as exc:
        print(f"fleet: {exc}", file=sys.stderr)
        return 1
    if not cases:
        print(f"fleet: no eval cases found in {fixtures_dir}", file=sys.stderr)
        return 1

    try:
        results = asyncio.run(run_eval(router, cases))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"fleet: eval failed — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    aggregates = aggregate(results)

    print(json.dumps({
        "n_cases": len(cases),
        "aggregates": aggregates,
    }, indent=2))

    if args.save_baseline:
        save_baseline(aggregates, args.save_baseline)
        print(f"\nbaseline saved to {args.save_baseline}", file=sys.stderr)

    if args.baseline:
        regressed, messages = compare_to_baseline(aggregates, args.baseline)
        print("\n--- comparison ---", file=sys.stderr)
        for m in messages:
            print(m, file=sys.stderr)
        if regressed:
            print("REGRESSION DETECTED", file=sys.stderr)
            return 3

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config = load_config(args.config)
    router = FleetRouter(config)

    if args.eval:
        return _run_eval(router, args)
    return _run_ask(router, args)


if __name__ == "__main__":
    sys.exit(main())

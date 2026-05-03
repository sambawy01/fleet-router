"""CLI entrypoint for direct testing outside Claude Code."""
from __future__ import annotations

import argparse
import asyncio
import sys

from fleet.config import load_config
from fleet.router import FleetRouter


def main() -> int:
    parser = argparse.ArgumentParser(description="Fleet Router CLI")
    parser.add_argument("prompt", nargs="+", help="Prompt text")
    parser.add_argument("--parallel", action="store_true", help="Force parallel mode")
    parser.add_argument("--model", default=None, help="Force specific model")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    router = FleetRouter(config)
    router._registry.refresh()
    prompt = " ".join(args.prompt)

    result = asyncio.run(router.ask(
        prompt,
        force_parallel=args.parallel,
        force_model=args.model,
    ))

    if isinstance(result, dict):
        for model, text in result.items():
            print(f"\n--- {model} ---\n{text}\n")
    else:
        print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())

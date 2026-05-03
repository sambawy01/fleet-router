"""Multi-provider dispatch with self-consistency support.

`EnsembleDispatcher.run` keeps the legacy 1-sample-per-model API for backward
compatibility. `run_multi` returns N samples per model for self-consistency
and judge-based synthesis.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fleet.config import Config
from fleet.providers.base import GenerateRequest, Provider
from fleet.providers.ollama import OllamaProvider
from fleet.providers.pool import ProviderPool

logger = logging.getLogger(__name__)


class EnsembleDispatcher:
    """Routes each model to its configured provider and dispatches in parallel."""

    def __init__(
        self,
        config: Config,
        pool: Optional[ProviderPool] = None,
    ):
        self._config = config
        self._pool = pool or ProviderPool.from_config(config)
        self._timeout = config.thresholds.parallel_timeout
        # Fallback provider for models that aren't in config (test compat,
        # ad-hoc CLI use). Always Ollama with the configured base_url.
        self._default_provider: Provider = (
            self._pool.get("ollama")
            or OllamaProvider(base_url=config.ollama.base_url, timeout=self._timeout)
        )

    async def run(
        self,
        prompt: str,
        models: list[str],
        system: Optional[str] = None,
    ) -> dict[str, Optional[str]]:
        """Single sample per model — backward-compat API."""
        multi = await self.run_multi(prompt, models, samples=1, system=system)
        return {m: (samples[0] if samples else None) for m, samples in multi.items()}

    async def run_multi(
        self,
        prompt: str,
        models: list[str],
        samples: int = 1,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> dict[str, list[str]]:
        """Returns each model's list of valid (non-None) samples. Empty list
        means every sample failed."""
        if not models:
            return {}
        plan: list[tuple[str, Provider, GenerateRequest]] = []
        for name in models:
            entry = self._config.models.get(name)
            if entry is not None:
                provider = self._pool.get(entry.provider) or self._default_provider
                api_model = entry.api_model or name
            else:
                provider = self._default_provider
                api_model = name
            req = GenerateRequest(
                model=api_model,
                prompt=prompt,
                system=system,
                temperature=temperature,
                samples=samples,
            )
            plan.append((name, provider, req))

        results = await asyncio.gather(
            *(provider.generate(req) for _, provider, req in plan),
            return_exceptions=True,
        )

        out: dict[str, list[str]] = {}
        for (name, _, _), result in zip(plan, results):
            if isinstance(result, BaseException):
                logger.warning("model %s dispatch crashed: %s", name, type(result).__name__)
                out[name] = []
            else:
                out[name] = [s for s in result if isinstance(s, str) and s]
        return out

    async def aclose(self) -> None:
        await self._pool.aclose_all()

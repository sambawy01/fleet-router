"""Retrieval augmentation — prepend grounded context to factual prompts.

Defines a RetrievalProvider Protocol and ships two implementations:
- `NoOpRetrieval`: returns no context (default; no external dependencies)
- `WebSearchRetrieval`: stub that demonstrates the integration shape; real
  implementations swap in SerpAPI / Bing / Tavily / DuckDuckGo as needed
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class RetrievalProvider(Protocol):
    name: str

    async def retrieve(self, query: str, max_chars: int) -> str:
        """Return a context string (possibly empty) to prepend to the prompt."""
        ...


class NoOpRetrieval:
    """No retrieval; useful as the default and for testing."""

    name = "noop"

    async def retrieve(self, query: str, max_chars: int) -> str:
        return ""


class WebSearchRetrieval:
    """Web search via a generic search API (SerpAPI-compatible).

    This is intentionally a thin scaffold — real deployments should
    customize the API endpoint and result parsing for their provider.
    Returns top-K result snippets concatenated with source URLs.
    """

    name = "websearch"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://serpapi.com/search.json",
        top_k: int = 5,
        timeout: int = 5,
    ):
        self._api_key = api_key or os.environ.get("SERP_API_KEY", "")
        self._endpoint = endpoint
        self._top_k = top_k
        self._timeout = timeout

    async def retrieve(self, query: str, max_chars: int) -> str:
        if not self._api_key:
            logger.warning("SERP_API_KEY missing; web search retrieval skipped")
            return ""
        # Lazy import keeps aiohttp out of the import-time path for users
        # who don't enable retrieval.
        import aiohttp
        params = {"q": query, "api_key": self._api_key, "num": self._top_k}
        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self._endpoint, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("web search failed: %s", exc)
            return ""

        results = data.get("organic_results") or []
        snippets: list[str] = []
        for r in results[: self._top_k]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            if not snippet:
                continue
            snippets.append(f"- {title}\n  {snippet}\n  Source: {link}")

        if not snippets:
            return ""
        context = "RETRIEVED CONTEXT:\n" + "\n\n".join(snippets)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n... [truncated]"
        return context


def build_retrieval_provider(name: str) -> RetrievalProvider:
    """Factory used by FleetRouter.__init__ based on config.retrieval.provider."""
    if name == "websearch":
        return WebSearchRetrieval()
    return NoOpRetrieval()

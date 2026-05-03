from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fleet.retrieval import (
    NoOpRetrieval,
    WebSearchRetrieval,
    build_retrieval_provider,
)


@pytest.mark.asyncio
async def test_noop_retrieval_returns_empty():
    r = NoOpRetrieval()
    assert await r.retrieve("anything", 1000) == ""


@pytest.mark.asyncio
async def test_websearch_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    r = WebSearchRetrieval(api_key="")
    assert await r.retrieve("query", 1000) == ""


@pytest.mark.asyncio
async def test_websearch_formats_results_when_key_present():
    r = WebSearchRetrieval(api_key="test-key", top_k=2)
    response_json = {
        "organic_results": [
            {"title": "Title One", "snippet": "First snippet", "link": "https://a.com"},
            {"title": "Title Two", "snippet": "Second snippet", "link": "https://b.com"},
        ]
    }
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=response_json)
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await r.retrieve("query", 1000)
    assert "Title One" in result
    assert "First snippet" in result
    assert "https://a.com" in result
    assert "Title Two" in result


@pytest.mark.asyncio
async def test_websearch_truncates_to_max_chars():
    r = WebSearchRetrieval(api_key="test-key")
    response_json = {
        "organic_results": [
            {"title": "T", "snippet": "x" * 5000, "link": "https://a.com"},
        ]
    }
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=response_json)
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await r.retrieve("q", max_chars=200)
    assert len(result) <= 200 + len("\n... [truncated]")
    assert result.endswith("... [truncated]")


def test_build_retrieval_provider_defaults_to_noop():
    p = build_retrieval_provider("anything")
    assert isinstance(p, NoOpRetrieval)


def test_build_retrieval_provider_websearch():
    p = build_retrieval_provider("websearch")
    assert isinstance(p, WebSearchRetrieval)

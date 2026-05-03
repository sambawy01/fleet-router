"""Ollama provider — talks to a local Ollama HTTP server."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

from fleet.providers.base import GenerateRequest, ModelInfo

logger = logging.getLogger(__name__)

_MAX_RESPONSE_CHARS = 4 * 1024 * 1024


class OllamaProvider:
    """Provider backed by Ollama's /api/generate."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        api_key: str = "",
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        """Return request headers including Authorization when api_key is set."""
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def generate(self, request: GenerateRequest) -> list[Optional[str]]:
        if request.samples < 1:
            return []
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        headers = self._headers()
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            tasks = [self._one_sample(session, request) for _ in range(request.samples)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[Optional[str]] = []
        for r in results:
            if isinstance(r, BaseException):
                logger.warning(
                    "ollama %s sample failed: %s", request.model, type(r).__name__
                )
                out.append(None)
            else:
                out.append(r)
        return out

    async def _one_sample(
        self,
        session: aiohttp.ClientSession,
        request: GenerateRequest,
    ) -> Optional[str]:
        payload: dict = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.system:
            payload["system"] = request.system
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens
        try:
            async with session.post(
                f"{self._base_url}/api/generate",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if not isinstance(data, dict) or "response" not in data:
                    logger.warning("missing 'response' from ollama %s", request.model)
                    return None
                response = data["response"]
                if not isinstance(response, str):
                    return None
                if len(response) > _MAX_RESPONSE_CHARS:
                    logger.warning(
                        "ollama %s response truncated to %d chars",
                        request.model, _MAX_RESPONSE_CHARS,
                    )
                    return response[:_MAX_RESPONSE_CHARS]
                return response
        except aiohttp.ClientResponseError as exc:
            logger.warning("HTTP %s from ollama %s", getattr(exc, "status", "?"), request.model)
            return None
        except aiohttp.ClientError as exc:
            logger.warning("client error from ollama %s: %s", request.model, type(exc).__name__)
            return None
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("timeout from ollama %s", request.model)
            return None

    async def list_models(self) -> list[ModelInfo]:
        timeout = aiohttp.ClientTimeout(total=5)
        headers = self._headers()
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(f"{self._base_url}/api/tags") as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ValueError) as exc:
            logger.warning("ollama list_models failed: %s", exc)
            return []
        models: list[ModelInfo] = []
        for entry in data.get("models", []) or []:
            if not isinstance(entry, dict):
                continue
            raw = entry.get("name")
            if not isinstance(raw, str) or not raw:
                continue
            models.append(ModelInfo(name=raw.split(":", 1)[0], provider=self.name))
        return models

    async def aclose(self) -> None:
        return None

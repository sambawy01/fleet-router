"""Discover Ollama models and map task tags to best model names."""
from __future__ import annotations

import requests
from fleet.config import Config


class ModelRegistry:
    """Knows which model handles which task."""

    def __init__(self, config: Config):
        self._config = config
        self._available: set[str] = set()
        self._refresh()

    def _refresh(self) -> None:
        """Fetch available Ollama models."""
        try:
            resp = requests.get(
                f"{self._config.ollama.base_url}/api/tags",
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            self._available = {
                m["name"].split(":")[0]  # strip tag
                for m in data.get("models", [])
            }
        except Exception:
            self._available = set(self._config.models.keys())

    def get_best_for_tag(self, tag: str) -> str | None:
        """Return single best model for a tag, or None if no match."""
        candidates = self.models_for_tag(tag, top_n=1)
        return candidates[0] if candidates else None

    def models_for_tag(self, tag: str, top_n: int = 3) -> list[str]:
        """Return top N models matching a tag, sorted by priority."""
        scored: list[tuple[str, int]] = []
        for name, entry in self._config.models.items():
            if tag in entry.tags and name in self._available:
                scored.append((name, entry.priority))
        scored.sort(key=lambda x: x[1])
        return [name for name, _ in scored[:top_n]]

    def all_available(self) -> list[str]:
        """List all currently available models."""
        return sorted(self._available)

"""Tag → Verifier mapping with fallback chain."""
from __future__ import annotations

from typing import Optional

from fleet.verifiers.base import Verifier
from fleet.verifiers.heuristic import HeuristicVerifier


class VerifierRegistry:
    """Resolves the right Verifier for a given task tag.

    Resolution order: explicit per-tag registration → default verifier →
    HeuristicVerifier(tag) as last-resort fallback.
    """

    def __init__(self, default: Optional[Verifier] = None):
        self._by_tag: dict[str, Verifier] = {}
        self._default = default

    def register(self, verifier: Verifier) -> None:
        self._by_tag[verifier.tag] = verifier

    def for_tag(self, tag: str) -> Verifier:
        explicit = self._by_tag.get(tag)
        if explicit is not None:
            return explicit
        if self._default is not None:
            return self._default
        return HeuristicVerifier(tag=tag)

    def has(self, tag: str) -> bool:
        return tag in self._by_tag

    def tags(self) -> list[str]:
        return sorted(self._by_tag.keys())

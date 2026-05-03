"""Verifier Protocol and shared dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Protocol, runtime_checkable


@dataclass
class Candidate:
    """A single sampled response from one model."""
    model: str
    sample_idx: int
    text: str
    score: float = 0.0
    notes: str = ""

    def with_score(self, score: float, notes: str = "") -> "Candidate":
        return replace(self, score=score, notes=notes or self.notes)


@dataclass
class VerificationResult:
    """Outcome of running a verifier across all candidates for one prompt.

    `abstain=True` signals "no candidate is good enough — return uncertainty
    structure instead of guessing." Callers (router) can choose to honor it
    or escalate to a stronger model with all candidates as context.
    """
    winner: Optional[Candidate]
    all_scored: list[Candidate]
    rationale: str = ""
    abstain: bool = False

    @property
    def winner_text(self) -> Optional[str]:
        return self.winner.text if self.winner else None


@runtime_checkable
class Verifier(Protocol):
    """Async tag-specific scorer + selector."""

    tag: str

    async def aggregate(
        self,
        prompt: str,
        candidates: list[Candidate],
    ) -> VerificationResult:
        ...


# Threshold below which `winner.score` triggers calibrated abstention.
DEFAULT_ABSTENTION_THRESHOLD = 0.4

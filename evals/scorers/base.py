"""Scorer protocol and shared dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class EvalCase:
    prompt: str
    tag: str
    expected: Any = None
    test_code: str = ""
    # Explicit scorer name override — empty falls back to the tag default.
    # Lets one tag (e.g. "reasoning") use multiple scorers across cases.
    scorer: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    case: EvalCase
    answer: str
    score: float  # 0.0 to 1.0
    notes: str = ""


@runtime_checkable
class Scorer(Protocol):
    name: str

    async def score(self, case: EvalCase, answer: str) -> EvalResult:
        ...

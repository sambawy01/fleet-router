"""Math scorer: extract the final numeric answer and compare to expected."""
from __future__ import annotations

import re

from evals.scorers.base import EvalCase, EvalResult

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_ANSWER_PATTERNS = [
    re.compile(r"(?:final\s+answer|answer|result)\s*[:=]?\s*\$?(-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"\\boxed\{([^}]+)\}"),
]


def _extract_final(text: str) -> str | None:
    text = _THINK_RE.sub("", text).strip()
    for pat in _ANSWER_PATTERNS:
        for m in pat.finditer(text):
            v = m.group(1).strip().replace(",", "").rstrip(".")
            if _NUMBER_RE.fullmatch(v):
                return v
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else None


class NumericMatchScorer:
    """Score = 1.0 if the answer's final number equals expected (within rel_tol)."""

    name = "numeric_match"

    def __init__(self, rel_tol: float = 1e-6):
        self._rel_tol = rel_tol

    async def score(self, case: EvalCase, answer: str) -> EvalResult:
        extracted = _extract_final(answer)
        if extracted is None:
            return EvalResult(case=case, answer=answer, score=0.0, notes="no number found")
        try:
            got = float(extracted)
            want = float(case.expected)
        except (TypeError, ValueError) as exc:
            return EvalResult(case=case, answer=answer, score=0.0, notes=f"parse error: {exc}")
        if want == 0:
            ok = abs(got) < self._rel_tol
        else:
            ok = abs(got - want) / abs(want) < self._rel_tol
        return EvalResult(
            case=case, answer=answer,
            score=1.0 if ok else 0.0,
            notes=f"got={got}, want={want}",
        )

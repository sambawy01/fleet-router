"""Multiple-choice scorer — extract A/B/C/D/E from a verbose LLM response."""
from __future__ import annotations

import re

from evals.scorers.base import EvalCase, EvalResult

# Strict markers first (prefer "the answer is X" over a stray "A" in the prose).
_ANSWER_PATTERNS = [
    re.compile(r"(?:final\s+answer|answer|correct\s+choice|choice)\s*(?:is\s*)?[:\-]?\s*\(?\b([A-E])\b\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-E])\)[\s.:]", re.MULTILINE),
    # "(X) is correct" / "X is the answer" — allow optional surrounding parens.
    re.compile(r"\(?\b([A-E])\)?\s*(?:is\s+correct|is\s+the\s+answer)", re.IGNORECASE),
]
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _extract_letter(text: str) -> str | None:
    text = _THINK_RE.sub("", text).strip()
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    # Last-resort: a letter on its own line near the end.
    for line in reversed(text.splitlines()):
        line = line.strip().rstrip(".")
        if re.fullmatch(r"\(?[A-E]\)?", line):
            return line.strip("()").upper()
    return None


class MultipleChoiceScorer:
    """Score = 1.0 iff the extracted letter matches `case.expected` (case-insensitive)."""

    name = "multi_choice"

    async def score(self, case: EvalCase, answer: str) -> EvalResult:
        got = _extract_letter(answer)
        if got is None:
            return EvalResult(case=case, answer=answer, score=0.0, notes="no letter found")
        want = str(case.expected).strip().upper()
        ok = got == want
        return EvalResult(
            case=case, answer=answer,
            score=1.0 if ok else 0.0,
            notes=f"got={got}, want={want}",
        )

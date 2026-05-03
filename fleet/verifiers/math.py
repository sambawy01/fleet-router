"""Math verifier — extracts numeric answers and majority-votes across candidates.

Self-consistency works particularly well here: when the same model is sampled
N times with temperature > 0, the majority numeric answer is usually correct
(Wang et al., 2022 — +18pp on GSM8K).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from fleet.verifiers.base import Candidate, VerificationResult

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# Match "the answer is X", "answer: X", "X" at end of line, etc.
_ANSWER_PATTERNS = [
    re.compile(r"(?:final\s+answer|answer|result|equals?|=)\s*[:=]?\s*\$?(-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"\\boxed\{([^}]+)\}"),
]
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _strip(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _extract_final_answer(text: str) -> Optional[str]:
    """Return the model's final numeric answer, normalized.

    Strategy: scan for explicit "answer is X" / \\boxed{X} markers first;
    fall back to the last number in the text (math models almost always
    end with their answer).
    """
    text = _strip(text)
    if not text:
        return None
    for pat in _ANSWER_PATTERNS:
        for m in pat.finditer(text):
            value = m.group(1).strip().replace(",", "").rstrip(".")
            if _NUMBER_RE.fullmatch(value):
                return _normalize(value)
    # Fallback: last number in text
    nums = _NUMBER_RE.findall(text)
    if nums:
        return _normalize(nums[-1])
    return None


def _normalize(value: str) -> str:
    """Canonicalize numeric strings so '42', '42.0', '4.2e1' compare equal."""
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return f"{f:g}"
    except (ValueError, OverflowError):
        return value


class MathVerifier:
    """Score by numeric answer; pick by majority vote with size-aware tie-break."""

    tag = "math"

    async def aggregate(
        self,
        prompt: str,
        candidates: list[Candidate],
    ) -> VerificationResult:
        if not candidates:
            return VerificationResult(winner=None, all_scored=[], rationale="no candidates", abstain=True)

        # Extract answers; keep candidates without parseable answers at score 0.
        per_candidate: list[tuple[Candidate, Optional[str]]] = []
        for c in candidates:
            ans = _extract_final_answer(c.text)
            per_candidate.append((c, ans))

        valid_answers = [a for _, a in per_candidate if a is not None]
        if not valid_answers:
            scored = [c.with_score(0.0, "no numeric answer found") for c, _ in per_candidate]
            return VerificationResult(
                winner=None, all_scored=scored,
                rationale="no candidate produced a parseable numeric answer",
                abstain=True,
            )

        votes = Counter(valid_answers)
        winner_answer, winner_count = votes.most_common(1)[0]
        agreement = winner_count / len(valid_answers)

        scored: list[Candidate] = []
        for c, ans in per_candidate:
            if ans is None:
                scored.append(c.with_score(0.0, "no answer"))
            elif ans == winner_answer:
                scored.append(c.with_score(agreement, f"answer={ans} ({winner_count}/{len(valid_answers)} agree)"))
            else:
                scored.append(c.with_score(0.2, f"answer={ans} (disagrees with majority {winner_answer})"))

        # Pick winner: among candidates with the majority answer, prefer the longest
        # explanation (proxy for show-your-work quality).
        winners = [c for c in scored if c.score == agreement]
        winner = max(winners, key=lambda c: len(c.text))

        # Abstain on tied vote with no clear majority on >2 distinct answers.
        abstain = agreement < 0.5 and len(votes) > 1
        return VerificationResult(
            winner=None if abstain else winner,
            all_scored=scored,
            rationale=f"majority answer: {winner_answer} ({winner_count}/{len(valid_answers)})",
            abstain=abstain,
        )

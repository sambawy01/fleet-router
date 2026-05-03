"""Text-match scorers for tags where exact-numeric or executable scoring
isn't applicable. Simple keyword-presence; for production use, swap in
ROUGE / BertScore / LLM-judge."""
from __future__ import annotations

import re

from evals.scorers.base import EvalCase, EvalResult


class KeywordContainsScorer:
    """Score = fraction of expected keywords present in the answer (case-insensitive).

    `case.expected` should be a list of keywords or a single string.
    """

    name = "keyword_contains"

    async def score(self, case: EvalCase, answer: str) -> EvalResult:
        expected = case.expected
        if isinstance(expected, str):
            keywords = [expected]
        elif isinstance(expected, list):
            keywords = [str(k) for k in expected]
        else:
            return EvalResult(case=case, answer=answer, score=0.0, notes="no expected keywords")
        if not keywords:
            return EvalResult(case=case, answer=answer, score=0.0, notes="empty keyword list")
        ans_lower = answer.lower()
        hits = sum(1 for k in keywords if k.lower() in ans_lower)
        score = hits / len(keywords)
        return EvalResult(
            case=case, answer=answer, score=score,
            notes=f"{hits}/{len(keywords)} keywords matched",
        )

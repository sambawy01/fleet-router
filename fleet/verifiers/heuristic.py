"""Heuristic verifier — wraps the original Synthesizer logic as a fallback
for when no judge / executable verifier is available."""
from __future__ import annotations

from fleet.synthesizer import Synthesizer
from fleet.verifiers.base import Candidate, VerificationResult


class HeuristicVerifier:
    """Adapter from rule-based Synthesizer.pick to the Verifier Protocol."""

    def __init__(self, tag: str = "general"):
        self.tag = tag
        self._synth = Synthesizer()

    async def aggregate(
        self,
        prompt: str,
        candidates: list[Candidate],
    ) -> VerificationResult:
        if not candidates:
            return VerificationResult(winner=None, all_scored=[], rationale="no candidates", abstain=True)

        # Group samples back per model for the legacy Synthesizer API:
        # {model: best_sample_text} — pick the longest sample per model since
        # the heuristic synthesizer expects 1 string per model.
        per_model: dict[str, Candidate] = {}
        for c in candidates:
            keep = per_model.get(c.model)
            if keep is None or len(c.text) > len(keep.text):
                per_model[c.model] = c

        responses = {model: c.text for model, c in per_model.items()}
        chosen = self._synth.pick(responses, task_tag=self.tag)

        if isinstance(chosen, dict):
            # Tie — heuristic returned all candidates; abstain so the router
            # can escalate or surface the tie to the user.
            scored = [c.with_score(0.5, "heuristic tie") for c in candidates]
            return VerificationResult(
                winner=None,
                all_scored=scored,
                rationale="heuristic synthesis returned a tie",
                abstain=True,
            )

        # Find the winning candidate by text match.
        scored: list[Candidate] = []
        winner: Candidate | None = None
        for c in candidates:
            if c.text == chosen and winner is None:
                w = c.with_score(0.7, f"heuristic[{self.tag}] winner")
                scored.append(w)
                winner = w
            else:
                scored.append(c.with_score(0.3, "heuristic non-winner"))
        return VerificationResult(
            winner=winner or scored[0],
            all_scored=scored,
            rationale=f"heuristic synthesizer picked for tag={self.tag}",
        )

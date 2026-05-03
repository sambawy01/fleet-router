"""Pick the best response from parallel model outputs."""
from __future__ import annotations

import difflib
import py_compile
import tempfile
from pathlib import Path
from typing import Optional


class Synthesizer:
    """Rule-based synthesis: no LLM, just heuristics."""

    def pick(self, responses: dict[str, Optional[str]], task_tag: str) -> str | dict[str, str]:
        """Return best response string, or dict of all if no clear winner."""
        # Filter out None/failed responses
        valid = {k: v for k, v in responses.items() if v}
        if not valid:
            return "(all models failed)"
        if len(valid) == 1:
            return next(iter(valid.values()))

        if task_tag == "code":
            return self._pick_code(valid)
        if task_tag in ("math", "reasoning"):
            return self._pick_reasoning(valid)
        if task_tag == "creative":
            return self._pick_creative(valid)
        if task_tag == "summarize":
            return self._pick_summarize(valid)
        return self._pick_general(valid)

    def _pick_code(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Prefer syntactically valid Python; longest among valid wins."""
        valid_python = [text for text in valid.values() if self._is_valid_python(text)]
        if valid_python:
            return max(valid_python, key=len)
        # Fallback: longest
        return max(valid.values(), key=len)

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                path = f.name
            py_compile.compile(path, doraise=True)
            Path(path).unlink(missing_ok=True)
            return True
        except Exception:
            Path(path).unlink(missing_ok=True)
            return False

    def _pick_reasoning(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Prefer consensus (similar answers)."""
        return self._consensus_or_longest(valid)

    def _pick_creative(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Prefer highest lexical diversity; tie-break with longest."""
        def diversity(text: str) -> float:
            words = text.split()
            if not words:
                return 0.0
            return len(set(words)) / len(words)

        return max(valid.values(), key=lambda t: (diversity(t), len(t)))

    def _pick_summarize(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Prefer shortest that still has content."""
        non_empty = [t for t in valid.values() if len(t.strip()) > 20]
        if non_empty:
            return min(non_empty, key=len)
        return min(valid.values(), key=len)

    def _pick_general(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Self-consistency: pick the answer most similar to others."""
        return self._consensus_or_longest(valid)

    def _consensus_or_longest(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Pick the response with highest average similarity to others.

        Self-consistency score is primary. If consensus is weak
        (best_score < 0.3), fall back to the longest response. If there is a
        tie for longest, return the full ``valid`` dict so the user can choose.
        """
        texts = list(valid.values())
        if len(texts) < 2:
            return texts[0]

        scores: dict[str, float] = {}
        for name, text in valid.items():
            sims = [
                difflib.SequenceMatcher(None, text, other).ratio()
                for other in texts if other != text
            ]
            scores[name] = sum(sims) / len(sims) if sims else 0.0

        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]

        if best_score >= 0.3:
            return valid[best_name]

        # Consensus is weak — fall back to longest response
        max_len = max(len(t) for t in texts)
        longest = [name for name, text in valid.items() if len(text) == max_len]
        if len(longest) == 1:
            return valid[longest[0]]
        return valid

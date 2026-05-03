"""Lightweight prompt classifier: keywords + optional embeddings."""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[misc]


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


# Keyword maps: tag → list of compiled regex patterns
KEYWORD_MAP: dict[str, list[re.Pattern[str]]] = {
    "code": _compile([
        r"\bpython\b", r"\bjavascript\b", r"\bjs\b", r"\btypescript\b", r"\bts\b",
        r"\bfunction\b", r"\bclass\b", r"\brefactor\b", r"\bdebug\b", r"\berror\b",
        r"\bcompile\b", r"\bsyntax\b", r"\bscript\b", r"\bmodule\b", r"\bimport\b",
        r"\bwrite\b.*\bcode\b", r"\bgenerate\b.*\bcode\b", r"\bunit test\b",
        r"\btest\b.*\bfunction\b", r"\bapi\b.*\bendpoint\b",
    ]),
    "math": _compile([
        r"\bcalculate\b", r"\bsolve\b", r"\bequation\b", r"\bmath\b", r"\bstatistics\b",
        r"\bprobability\b", r"\bderivative\b", r"\bintegral\b", r"\bformula\b",
        r"\bsum\b.*\bnumbers\b", r"\bmultiply\b", r"\bdivide\b",
    ]),
    "reasoning": _compile([
        r"\bexplain\b.*\bwhy\b", r"\bcompare\b", r"\bcontrast\b", r"\banalyze\b",
        r"\bevaluate\b", r"\bpros?\b.*\bcons?\b", r"\badvantages?\b", r"\bdisadvantages?\b",
        r"\bwhat\b.*\bif\b", r"\bshould\b.*\bchoose\b",
    ]),
    "creative": _compile([
        r"\bpoem\b", r"\bstory\b", r"\bjoke\b", r"\btagline\b", r"\bslogan\b",
        r"\bcreative\b", r"\bimagine\b", r"\bdesign\b", r"\bbrainstorm\b",
        r"\bwrite\b.*\bstory\b", r"\bdraft\b", r"\bhook\b", r"\bcaption\b",
    ]),
    "summarize": _compile([
        r"\bsummarize\b", r"\bsummary\b", r"\btl;dr\b", r"\bkey points\b",
        r"\bmain ideas?\b", r"\brecap\b", r"\bcondense\b",
    ]),
    "translate": _compile([
        r"\btranslate\b", r"\btranslation\b", r"\bchinese\b", r"\barabic\b",
        r"\bspanish\b", r"\bfrench\b", r"\bgerman\b", r"\bjapanese\b",
    ]),
}

# Uncertainty keywords that bias toward parallel mode
UNCERTAINTY_MARKERS: list[re.Pattern[str]] = _compile([
    r"\bbest\b", r"\bcompare\b", r"\breview\b", r"\bimprove\b",
    r"\boptimize\b", r"\bwhich\b.*\bbetter\b", r"\bwhat\b.*\bthink\b",
    r"\bsuggest\b", r"\brecommend\b",
])

# Saturating per-tag confidence as a function of keyword match count.
# 1 match → 0.55, 2 → 0.80, 3 → 0.91, 4 → 0.96. Single accidental hits
# (e.g. "I had an error yesterday") stay below the single_confidence
# threshold so the router falls back to parallel mode.
_SATURATION_BASE = 0.45

# Cosine threshold below which the embedding signal is treated as noise.
_EMBED_RELEVANCE_FLOOR = 0.30
# Multiplier applied to the dominant tag's similarity when blending into
# the keyword score; small enough to refine, not override.
_EMBED_BONUS_WEIGHT = 0.20


class TaskClassifier:
    """Classify a prompt into a task tag and confidence score."""

    def __init__(self, embeddings_model: Optional[str] = None):
        self._embeddings_model = embeddings_model
        self._model = None
        self._tag_embeddings: Optional[dict] = None

        if embeddings_model:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(embeddings_model)
                self._tag_embeddings = {
                    tag: self._model.encode(
                        f"Task: {tag}. {self._tag_description(tag)}"
                    )
                    for tag in KEYWORD_MAP
                }
            except Exception as exc:  # noqa: BLE001 — graceful degradation
                logger.warning(
                    "embeddings model %r unavailable (%s); using keywords only",
                    embeddings_model, exc,
                )
                self._model = None
                self._tag_embeddings = None

    @staticmethod
    def _tag_description(tag: str) -> str:
        descriptions = {
            "code": "writing programming code and software development",
            "math": "mathematical calculations and solving equations",
            "reasoning": "logical reasoning and analysis",
            "creative": "creative writing and storytelling",
            "summarize": "summarizing text and extracting key points",
            "translate": "translating between languages",
        }
        return descriptions.get(tag, "general task")

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        if np is None:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def classify(self, prompt: str) -> tuple[str, float]:
        prompt_lower = prompt.lower()

        # 1. Keyword scoring — saturating exponential so a single accidental
        # match cannot trip the single-model confidence threshold.
        scores: dict[str, float] = {}
        for tag, patterns in KEYWORD_MAP.items():
            matches = sum(1 for p in patterns if p.search(prompt_lower))
            scores[tag] = (1.0 - _SATURATION_BASE ** matches) if matches > 0 else 0.0

        # 2. Uncertainty penalty (capped at 0.4)
        uncertainty = sum(1 for p in UNCERTAINTY_MARKERS if p.search(prompt_lower))
        uncertainty_penalty = min(uncertainty * 0.15, 0.4)

        # 3. Embedding refinement — bonus only to the embedding-preferred tag,
        # only when cosine similarity clears a relevance floor. Refines the
        # keyword score; never overrides a strong keyword signal alone.
        if self._model is not None and self._tag_embeddings is not None:
            try:
                prompt_emb = self._model.encode(prompt)
                sims = {
                    tag: self._cosine_similarity(prompt_emb, tag_emb)
                    for tag, tag_emb in self._tag_embeddings.items()
                }
                best_emb_tag = max(sims, key=sims.get)
                if sims[best_emb_tag] >= _EMBED_RELEVANCE_FLOOR:
                    scores[best_emb_tag] = min(
                        1.0,
                        scores.get(best_emb_tag, 0.0)
                        + sims[best_emb_tag] * _EMBED_BONUS_WEIGHT,
                    )
            except Exception as exc:  # noqa: BLE001 — embeddings are best-effort
                logger.warning("embedding classify failed: %s", exc)

        if not any(scores.values()):
            return "general", 0.0

        best_tag = max(scores, key=scores.get)
        best_score = scores[best_tag]
        confidence = max(0.0, best_score - uncertainty_penalty)
        return best_tag, confidence

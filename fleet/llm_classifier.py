"""LLM-based prompt classifier — sibling to TaskClassifier.

Sends the prompt + the list of tags to a small instruct model and parses
back a structured `(tag, confidence, reasoning)` triple. Falls back to
TaskClassifier (keyword) when the LLM call fails or returns garbage.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from fleet.classifier import TaskClassifier
from fleet.providers.base import GenerateRequest, Provider

logger = logging.getLogger(__name__)

_VALID_TAGS = ("code", "math", "reasoning", "creative", "summarize", "translate", "general")

_PROMPT = """Classify the user prompt into one task tag from this list:
- code: programming, debugging, refactoring
- math: arithmetic, algebra, calculus, statistics
- reasoning: explanation, comparison, analysis, logical argument
- creative: poems, stories, copywriting, brainstorming
- summarize: condensing existing text, extracting key points
- translate: translating between human languages
- general: anything else

USER PROMPT:
{prompt}

Reply with ONLY this JSON object:
{{"tag": "<one of the above>", "confidence": 0.0-1.0, "reasoning": "brief"}}"""


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    start = -1
    return None


class LLMClassifier:
    """Zero-shot classification via a small instruct model."""

    def __init__(
        self,
        provider: Provider,
        model: str,
        fallback: Optional[TaskClassifier] = None,
        temperature: float = 0.0,
    ):
        self._provider = provider
        self._model = model
        self._fallback = fallback or TaskClassifier()
        self._temperature = temperature

    async def classify(self, prompt: str) -> tuple[str, float]:
        req = GenerateRequest(
            model=self._model,
            prompt=_PROMPT.format(prompt=prompt),
            temperature=self._temperature,
            samples=1,
        )
        try:
            results = await self._provider.generate(req)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM classifier provider crashed: %s; using fallback", exc)
            return self._fallback.classify(prompt)

        if not results or not results[0]:
            logger.warning("LLM classifier produced no output; using fallback")
            return self._fallback.classify(prompt)

        parsed = _extract_json(results[0])
        if not parsed or not isinstance(parsed, dict):
            logger.warning("LLM classifier output unparseable; using fallback")
            return self._fallback.classify(prompt)

        tag = str(parsed.get("tag", "")).strip().lower()
        if tag not in _VALID_TAGS:
            logger.warning("LLM classifier returned unknown tag %r; using fallback", tag)
            return self._fallback.classify(prompt)

        try:
            conf = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        return tag, conf

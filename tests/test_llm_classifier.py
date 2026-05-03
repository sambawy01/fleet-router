from unittest.mock import AsyncMock

import pytest

from fleet.classifier import TaskClassifier
from fleet.llm_classifier import LLMClassifier


@pytest.mark.asyncio
async def test_llm_classifier_parses_structured_output():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=[
        '{"tag": "code", "confidence": 0.92, "reasoning": "function keyword"}'
    ])
    c = LLMClassifier(provider=provider, model="qwen-tiny")
    tag, conf = await c.classify("write a function that sorts")
    assert tag == "code"
    assert conf == 0.92


@pytest.mark.asyncio
async def test_llm_classifier_falls_back_on_unknown_tag():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=['{"tag": "alien", "confidence": 1.0}'])
    fallback = TaskClassifier()
    c = LLMClassifier(provider=provider, model="qwen-tiny", fallback=fallback)
    tag, conf = await c.classify("write a python function")
    # Fallback resolves to "code" via keyword path.
    assert tag == "code"


@pytest.mark.asyncio
async def test_llm_classifier_falls_back_on_unparseable_output():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=["lol just text"])
    fallback = TaskClassifier()
    c = LLMClassifier(provider=provider, model="qwen-tiny", fallback=fallback)
    tag, _ = await c.classify("write a python function")
    assert tag == "code"


@pytest.mark.asyncio
async def test_llm_classifier_falls_back_on_provider_exception():
    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=RuntimeError("boom"))
    c = LLMClassifier(provider=provider, model="qwen-tiny")
    tag, _ = await c.classify("write a python function")
    assert tag == "code"  # keyword fallback


@pytest.mark.asyncio
async def test_llm_classifier_clips_confidence():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=['{"tag": "code", "confidence": 99}'])
    c = LLMClassifier(provider=provider, model="qwen-tiny")
    _, conf = await c.classify("p")
    assert conf == 1.0

"""Tests for fleet.proxy — Anthropic Messages API compatibility layer."""
from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from fleet.proxy import (
    _flatten_content,
    _maybe_enrich_with_ollama_hint,
    _parse_request,
    build_app,
)
from fleet.router import ERROR_ALL_MODELS_FAILED, ERROR_NO_MODEL


class _StubRouter:
    """Stand-in for FleetRouter.ask — captures the prompt + system it saw."""

    def __init__(self, answer: str = "hello back"):
        self._answer = answer
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    async def ask(self, prompt, *, force_parallel=False, force_model=None, system=None):
        self.last_prompt = prompt
        self.last_system = system
        return self._answer


# ---------- pure unit tests (no HTTP) ----------

def test_flatten_content_string_passthrough():
    assert _flatten_content("hi") == "hi"


def test_flatten_content_text_blocks():
    blocks = [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    assert _flatten_content(blocks) == "one\ntwo"


def test_flatten_content_tool_use_summarized():
    blocks = [{"type": "tool_use", "name": "Read", "input": {"path": "/tmp/x"}}]
    out = _flatten_content(blocks)
    assert "tool_call" in out and "Read" in out and "/tmp/x" in out


def test_flatten_content_tool_result_recurses():
    blocks = [{
        "type": "tool_result",
        "tool_use_id": "tool_123",
        "content": [{"type": "text", "text": "result body"}],
    }]
    out = _flatten_content(blocks)
    assert "tool_result" in out and "tool_123" in out and "result body" in out


def test_parse_request_collapses_history_with_role_markers():
    body = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
        "max_tokens": 100,
    }
    parsed = _parse_request(body)
    assert "Human: first question" in parsed.prompt
    assert "Assistant: first answer" in parsed.prompt
    assert "Human: second question" in parsed.prompt
    assert parsed.prompt.rstrip().endswith("Assistant:")
    assert parsed.requested_model == "claude-3-5-sonnet"
    assert parsed.stream is False


def test_parse_request_carries_system_string():
    body = {
        "model": "x",
        "system": "you are concise",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert _parse_request(body).system == "you are concise"


def test_parse_request_flattens_system_blocks():
    body = {
        "model": "x",
        "system": [{"type": "text", "text": "rule one"}, {"type": "text", "text": "rule two"}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert _parse_request(body).system == "rule one\nrule two"


def test_parse_request_rejects_empty_messages():
    with pytest.raises(web.HTTPBadRequest):
        _parse_request({"model": "x", "messages": []})


# ---------- HTTP integration tests ----------

@pytest.mark.asyncio
async def test_messages_non_streaming_shape():
    router = _StubRouter("the answer is 42")
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "what is the answer"}],
            "max_tokens": 100,
        })
        assert resp.status == 200
        body = await resp.json()

    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-3-5-sonnet"
    assert body["stop_reason"] == "end_turn"
    assert body["content"] == [{"type": "text", "text": "the answer is 42"}]
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] > 0
    assert body["id"].startswith("msg_")
    assert "Human: what is the answer" in (router.last_prompt or "")


@pytest.mark.asyncio
async def test_messages_streaming_event_sequence():
    router = _StubRouter("streamed text payload")
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "stream please"}],
            "max_tokens": 100,
            "stream": True,
        })
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        raw = (await resp.read()).decode("utf-8")

    # Parse the SSE event sequence.
    events = [
        line[len("event: "):].strip()
        for line in raw.splitlines() if line.startswith("event: ")
    ]
    assert events[0] == "message_start"
    assert events[1] == "content_block_start"
    assert "content_block_delta" in events
    assert events[-3] == "content_block_stop"
    assert events[-2] == "message_delta"
    assert events[-1] == "message_stop"

    # Reassemble the streamed text from data: lines.
    deltas = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line[len("data: "):])
            if payload.get("type") == "content_block_delta":
                deltas.append(payload["delta"]["text"])
    assert "".join(deltas) == "streamed text payload"


@pytest.mark.asyncio
async def test_api_key_required_when_configured():
    router = _StubRouter()
    app = build_app(router, api_key="secret-token")  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        # No header — rejected.
        bad = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert bad.status == 401

        # Correct header — accepted.
        ok = await client.post(
            "/v1/messages",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "secret-token"},
        )
        assert ok.status == 200


@pytest.mark.asyncio
async def test_healthz():
    app = build_app(_StubRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/healthz")
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True


@pytest.mark.asyncio
async def test_router_failure_returns_500():
    class _BoomRouter:
        async def ask(self, *args, **kwargs):
            raise RuntimeError("ollama unreachable")

    app = build_app(_BoomRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status == 500


# ---------- Ollama-down hint enrichment ----------

def test_maybe_enrich_passes_through_normal_text():
    assert _maybe_enrich_with_ollama_hint("hello world") == "hello world"


def test_maybe_enrich_passes_through_unrelated_parenthesised():
    # Strings that start with "(" but aren't router sentinels stay untouched.
    assert _maybe_enrich_with_ollama_hint("(this is just a parenthetical)") == \
        "(this is just a parenthetical)"


def test_maybe_enrich_appends_hint_to_all_models_failed_sentinel():
    out = _maybe_enrich_with_ollama_hint(ERROR_ALL_MODELS_FAILED)
    assert out.startswith(ERROR_ALL_MODELS_FAILED)
    assert "ollama serve" in out


def test_maybe_enrich_appends_hint_to_no_model_sentinel_with_suffix():
    text = f"{ERROR_NO_MODEL} for tag: code"
    out = _maybe_enrich_with_ollama_hint(text)
    assert out.startswith(text)
    assert "ollama serve" in out


@pytest.mark.asyncio
async def test_messages_enriches_ollama_down_response():
    """When router.ask returns an error sentinel (Ollama down), the proxy
    must return 200 with the sentinel + actionable troubleshooting text —
    NOT a raw HTTP error, since errors mid-stream can't be recovered cleanly."""
    router = _StubRouter(ERROR_ALL_MODELS_FAILED)
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status == 200
        body = await resp.json()
    text = body["content"][0]["text"]
    assert text.startswith(ERROR_ALL_MODELS_FAILED)
    assert "ollama serve" in text


@pytest.mark.asyncio
async def test_messages_streaming_enriches_ollama_down_response():
    router = _StubRouter(ERROR_ALL_MODELS_FAILED)
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status == 200
        raw = (await resp.read()).decode("utf-8")
    deltas = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line[len("data: "):])
            if payload.get("type") == "content_block_delta":
                deltas.append(payload["delta"]["text"])
    full_text = "".join(deltas)
    assert ERROR_ALL_MODELS_FAILED in full_text
    assert "ollama serve" in full_text


# ---------- /v1/models endpoint ----------

@pytest.mark.asyncio
async def test_v1_models_returns_openai_shape():
    """Stub router has no _registry attribute — endpoint must degrade gracefully."""
    router = _StubRouter()
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/v1/models")
        assert resp.status == 200
        body = await resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    # No _registry on stub → empty list, not a 500.
    assert body["data"] == []


@pytest.mark.asyncio
async def test_v1_models_returns_registry_models():
    class _RegistryStub:
        def all_available(self):
            return ["deepseek-v4-pro", "glm-5.1"]

    class _RouterWithRegistry:
        def __init__(self):
            self._registry = _RegistryStub()

        async def ask(self, *args, **kwargs):
            return "n/a"

    app = build_app(_RouterWithRegistry())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/v1/models")
        body = await resp.json()
    ids = [m["id"] for m in body["data"]]
    assert ids == ["deepseek-v4-pro", "glm-5.1"]
    assert all(m["object"] == "model" and m["owned_by"] == "fleet" for m in body["data"])

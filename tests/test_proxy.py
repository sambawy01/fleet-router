"""Tests for fleet.proxy — Anthropic Messages API compatibility layer."""
from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from fleet.proxy import _flatten_content, _parse_request, build_app


class _StubRouter:
    """Stand-in for FleetRouter.ask — captures the prompt + system it saw."""

    def __init__(self, answer: str = "hello back", delay: float = 0.0):
        self._answer = answer
        self._delay = delay
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.call_count = 0

    async def ask(self, prompt, *, force_parallel=False, force_model=None, system=None):
        self.last_prompt = prompt
        self.last_system = system
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
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


# ---------- streaming heartbeat / deadline / concurrency ----------


def _parse_sse_events(raw: str) -> list[tuple[str, dict]]:
    """Parse an SSE byte stream into (event_name, data_dict) pairs."""
    events: list[tuple[str, dict]] = []
    current_event: str | None = None
    for line in raw.splitlines():
        if line.startswith("event: "):
            current_event = line[len("event: "):].strip()
        elif line.startswith("data: ") and current_event is not None:
            try:
                payload = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                payload = {}
            events.append((current_event, payload))
            current_event = None
    return events


@pytest.mark.asyncio
async def test_streaming_emits_message_start_before_router_finishes():
    """Regression guard for the v1 'fake streaming' bug — the v1 proxy
    awaited router.ask() FULLY before opening SSE, so Claude Code saw
    silence for the entire synthesis window. v2 must emit message_start
    immediately and ping while waiting."""
    # 1.5s delay: long enough that we'll see at least one ping at 5s
    # heartbeat... too long for fast CI. Use shorter heartbeat for the
    # test so we don't have to wait actual seconds.
    import fleet.proxy as proxy_mod
    original_heartbeat = proxy_mod._HEARTBEAT_INTERVAL_S
    proxy_mod._HEARTBEAT_INTERVAL_S = 0.1
    try:
        router = _StubRouter("done", delay=0.35)  # ~3 ping cycles
        app = build_app(router)  # type: ignore[arg-type]
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/v1/messages", json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
            assert resp.status == 200
            raw = (await resp.read()).decode("utf-8")
    finally:
        proxy_mod._HEARTBEAT_INTERVAL_S = original_heartbeat

    events = _parse_sse_events(raw)
    event_names = [name for name, _ in events]
    assert event_names[0] == "message_start"
    # At least one ping must have fired between message_start and the
    # actual content — otherwise streaming is "fake" again.
    assert "ping" in event_names, f"no heartbeat events; got {event_names}"
    ping_idx = event_names.index("ping")
    content_idx = event_names.index("content_block_start")
    assert ping_idx < content_idx, "ping should fire BEFORE content arrives"
    assert event_names[-1] == "message_stop"


@pytest.mark.asyncio
async def test_streaming_router_failure_yields_clean_stream_close():
    """If router.ask raises after headers are out, we can't switch to a
    500 — must surface the error inside the SSE body and close cleanly."""
    class _BoomRouter:
        async def ask(self, *args, **kwargs):
            raise RuntimeError("ollama crashed mid-prompt")

    app = build_app(_BoomRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status == 200
        raw = (await resp.read()).decode("utf-8")

    events = _parse_sse_events(raw)
    names = [n for n, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    # The error text should be in a content_block_delta.
    deltas = [
        d["delta"]["text"] for n, d in events
        if n == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
    ]
    assert any("router error" in t and "ollama crashed mid-prompt" in t for t in deltas)


@pytest.mark.asyncio
async def test_prompt_deadline_non_streaming_returns_504():
    """A router.ask that exceeds prompt_deadline_s on the non-streaming
    path must surface as 504, not as a 60s+ silent hang."""
    router = _StubRouter("never seen", delay=10.0)
    app = build_app(router, prompt_deadline_s=0.2)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status == 504


@pytest.mark.asyncio
async def test_prompt_deadline_streaming_closes_with_error_block():
    """Streaming path: deadline exceeded should close cleanly with an
    error message-block, not hang the connection."""
    import fleet.proxy as proxy_mod
    original_heartbeat = proxy_mod._HEARTBEAT_INTERVAL_S
    proxy_mod._HEARTBEAT_INTERVAL_S = 0.05
    try:
        router = _StubRouter("never seen", delay=10.0)
        app = build_app(router, prompt_deadline_s=0.2)  # type: ignore[arg-type]
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/v1/messages", json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
            assert resp.status == 200
            raw = (await resp.read()).decode("utf-8")
    finally:
        proxy_mod._HEARTBEAT_INTERVAL_S = original_heartbeat
    events = _parse_sse_events(raw)
    names = [n for n, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    # Error surfaces in the stream as text content.
    deltas = [
        d["delta"]["text"] for n, d in events
        if n == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
    ]
    joined = "".join(deltas)
    assert "deadline" in joined.lower() or "timeout" in joined.lower()


@pytest.mark.asyncio
async def test_concurrent_proxy_requests_all_complete():
    """N parallel /v1/messages calls must all return their own answers
    without any cross-talk or session reuse bugs."""
    router = _StubRouter("answer", delay=0.05)
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        async def one_request(i):
            resp = await client.post("/v1/messages", json={
                "model": "x",
                "messages": [{"role": "user", "content": f"prompt {i}"}],
            })
            assert resp.status == 200
            body = await resp.json()
            return body["content"][0]["text"]

        results = await asyncio.gather(*(one_request(i) for i in range(20)))
    assert all(r == "answer" for r in results)
    assert router.call_count == 20

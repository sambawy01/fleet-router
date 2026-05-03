import json

from fleet.events import (
    EventBus,
    JSONLSink,
    ModelDispatched,
    PromptClassified,
    ResponseSynthesized,
    cli_progress_sink,
    logging_sink,
)


def test_subscriber_receives_events():
    received = []
    bus = EventBus()
    bus.subscribe(lambda e: received.append(e))
    bus.emit(PromptClassified(tag="code", confidence=0.9))
    bus.emit(ResponseSynthesized(tag="code", mode="verifier"))
    assert len(received) == 2
    assert received[0].tag == "code"


def test_sink_exception_swallowed():
    bus = EventBus()
    bus.subscribe(lambda e: 1 / 0)  # raises
    received = []
    bus.subscribe(lambda e: received.append(e))
    bus.emit(PromptClassified())  # should not raise
    assert len(received) == 1


def test_no_subscribers_emit_is_no_op():
    bus = EventBus()
    bus.emit(PromptClassified())  # no error


def test_jsonl_sink_writes_one_line_per_event(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = JSONLSink(str(path))
    sink(PromptClassified(tag="code", confidence=0.9, prompt="write a fn"))
    sink(ResponseSynthesized(tag="code", mode="verifier", winner_model="glm"))

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["event"] == "PromptClassified"
    assert parsed[0]["tag"] == "code"
    assert parsed[1]["winner_model"] == "glm"


def test_logging_sink_does_not_crash(caplog):
    logging_sink(PromptClassified(tag="x"))  # smoke test


def test_cli_progress_sink_emits_one_line_per_event(capsys):
    cli_progress_sink(PromptClassified(tag="creative", confidence=0.42))
    cli_progress_sink(ModelDispatched(
        models=["glm-5.1", "deepseek-v4-pro"], tag="creative", samples=5,
    ))
    cli_progress_sink(ResponseSynthesized(
        tag="creative", mode="verifier",
        winner_model="glm-5.1", winner_score=0.78,
    ))
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line]
    assert len(lines) == 3
    assert "classified as 'creative'" in lines[0] and "0.42" in lines[0]
    assert "dispatching 2 model(s) × 5 sample(s)" in lines[1]
    assert "glm-5.1" in lines[1] and "deepseek-v4-pro" in lines[1]
    assert "synthesized [verifier]" in lines[2]
    assert "winner=glm-5.1" in lines[2] and "0.78" in lines[2]


def test_cli_progress_sink_handles_abstain_and_missing_score(capsys):
    cli_progress_sink(ResponseSynthesized(
        tag="reasoning", mode="verifier", abstain=True,
    ))
    cli_progress_sink(ResponseSynthesized(
        tag="reasoning", mode="heuristic",
    ))
    err = capsys.readouterr().err.splitlines()
    assert "abstained" in err[0]
    # No winner, no abstain → falls back to mode label.
    assert "heuristic" in err[1]


def test_cli_progress_sink_ignores_unknown_event_types():
    """Sink should silently skip event types it doesn't render — no exception."""
    from fleet.events import RouterEvent
    cli_progress_sink(RouterEvent())  # base event, no formatter — must not raise

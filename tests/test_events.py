import json

from fleet.events import (
    EventBus,
    JSONLSink,
    PromptClassified,
    ResponseSynthesized,
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

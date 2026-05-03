import pytest

from evals.compare import compare


class _StubRouter:
    def __init__(self, answers):
        self._answers = answers
    async def ask(self, prompt):
        return self._answers.get(prompt, "no answer")


@pytest.mark.asyncio
async def test_compare_reports_per_tag_deltas(tmp_path):
    f = tmp_path / "math.jsonl"
    f.write_text(
        '{"prompt": "1+1?", "tag": "math", "expected": 2}\n'
        '{"prompt": "2+2?", "tag": "math", "expected": 4}\n'
    )
    router_a = _StubRouter({"1+1?": "the answer is 2", "2+2?": "I think 99"})
    router_b = _StubRouter({"1+1?": "the answer is 2", "2+2?": "the answer is 4"})

    report = await compare(
        a=("a", router_a),
        b=("b", router_b),
        fixtures_dir=tmp_path,
    )
    assert report["aggregates"]["a"]["math"]["pass_rate"] == 0.5
    assert report["aggregates"]["b"]["math"]["pass_rate"] == 1.0
    assert "delta=+50.0pp" in report["summary"]
    assert "→ b" in report["summary"]


@pytest.mark.asyncio
async def test_compare_handles_router_exception(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"prompt": "p", "tag": "math", "expected": 1}\n')

    class Crashing:
        async def ask(self, prompt):
            raise RuntimeError("network down")

    report = await compare(
        a=("crash", Crashing()),
        b=("ok", _StubRouter({"p": "1"})),
        fixtures_dir=tmp_path,
    )
    # Crash recorded as score=0 (extracted no number from error message)
    assert report["aggregates"]["crash"]["math"]["pass_rate"] == 0.0
    assert report["aggregates"]["ok"]["math"]["pass_rate"] == 1.0

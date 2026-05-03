import pytest

from fleet.verifiers.base import Candidate
from fleet.verifiers.code import CodeVerifier


@pytest.mark.asyncio
async def test_code_verifier_picks_parseable_over_garbage():
    v = CodeVerifier()
    candidates = [
        Candidate("a", 0, "this is not python ("),
        Candidate("b", 0, "def f():\n    return 1"),
    ]
    result = await v.aggregate("prompt", candidates)
    assert result.winner is not None
    assert result.winner.model == "b"
    assert not result.abstain


@pytest.mark.asyncio
async def test_code_verifier_strips_fenced_blocks():
    v = CodeVerifier()
    candidates = [
        Candidate("a", 0, "Here is the code:\n```python\ndef foo():\n    return 42\n```\nLet me know."),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert "def foo" in result.winner.text


@pytest.mark.asyncio
async def test_code_verifier_abstains_when_nothing_parses():
    v = CodeVerifier()
    candidates = [
        Candidate("a", 0, "definitely not code ((("),
        Candidate("b", 0, "more not code )))"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.abstain
    assert result.winner is None


@pytest.mark.asyncio
async def test_code_verifier_refuses_to_execute_dangerous_code():
    """Even with execute=True, statically-detectable dangerous patterns must
    not run. Score caps at 0.5 (parses) without an execution bonus."""
    v = CodeVerifier(execute=True)
    candidates = [
        Candidate("a", 0, "import os\nos.system('echo hi')"),
    ]
    result = await v.aggregate("p", candidates)
    # Parses, but flagged as unsafe — score should be ≤ 0.5 (no exec bonus).
    assert result.winner is not None
    assert result.winner.score <= 0.5
    assert "unsafe" in result.winner.notes


@pytest.mark.asyncio
async def test_code_verifier_executes_safe_code():
    v = CodeVerifier(execute=True, execute_timeout=10)
    candidates = [
        Candidate("a", 0, "x = 1 + 1\nassert x == 2"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.score == 1.0
    assert "executes cleanly" in result.winner.notes


@pytest.mark.asyncio
async def test_code_verifier_handles_empty_input():
    v = CodeVerifier()
    result = await v.aggregate("p", [])
    assert result.abstain

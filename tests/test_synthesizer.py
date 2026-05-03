import pytest
from fleet.synthesizer import Synthesizer


def test_pick_code_valid():
    synth = Synthesizer()
    responses = {
        "deepseek": "def foo():\n    return 1",
        "glm": "def bar():\n    x = 1\n    return x + 2",
        "minimax": "invalid python here",
    }
    best = synth.pick(responses, task_tag="code")
    # Both deepseek and glm are valid; glm is longest
    assert best == "def bar():\n    x = 1\n    return x + 2"


def test_pick_code_no_valid():
    synth = Synthesizer()
    responses = {
        "deepseek": "broken syntax (",
        "glm": "also broken [ but longer now",
    }
    best = synth.pick(responses, task_tag="code")
    assert best == "also broken [ but longer now"


def test_pick_creative_diversity():
    synth = Synthesizer()
    responses = {
        "glm": "word word word word word word word",
        "minimax": "a much longer and more detailed creative response with many words",
    }
    best = synth.pick(responses, task_tag="creative")
    # minimax has higher lexical diversity (more unique words relative to length)
    assert "much longer" in best


def test_pick_creative_tie_breaker():
    synth = Synthesizer()
    responses = {
        "glm": "alpha beta gamma delta epsilon",
        "minimax": "uno dos tres cuatro cinco",
    }
    best = synth.pick(responses, task_tag="creative")
    # Both have diversity 1.0 and equal length — tie-break is stable on the
    # first dict insertion. Result must be a string from the candidate set.
    assert isinstance(best, str)
    assert best in responses.values()


def test_pick_code_strips_fenced_block_before_parsing():
    """ast.parse must ignore prose wrapped around ```python ... ``` fences."""
    synth = Synthesizer()
    responses = {
        "glm": "Here is the code:\n```python\ndef foo():\n    return 1\n```\nLet me know if you want changes.",
        "minimax": "this won't parse at all (",
    }
    best = synth.pick(responses, task_tag="code")
    # glm wraps valid Python in a fence; should be picked over minimax.
    assert "def foo" in best


def test_pick_strips_thinking_tokens_from_output():
    """<think>...</think> chain-of-thought blocks must be stripped from the
    returned answer so users don't see internal reasoning."""
    synth = Synthesizer()
    responses = {
        "reasoning-model": "<think>Let me think...\nThe user wants 42.</think>The answer is 42.",
        "chat-model": "the answer is 42",
    }
    best = synth.pick(responses, task_tag="general")
    # Must not contain the thinking block, must contain the actual answer
    assert "<think>" not in best
    assert "Let me think" not in best
    assert "42" in best


def test_pick_thinking_tokens_dont_inflate_length_score():
    """A reasoning model dumping 5KB of <think> shouldn't beat a concise
    chat-model answer just because the response field is longer."""
    synth = Synthesizer()
    huge_thinking = "reasoning step. " * 500  # ~7KB
    responses = {
        "reasoning-model": f"<think>{huge_thinking}</think>short final answer",
        "chat-model": "this is a much longer concrete answer with many more characters than the reasoning model's stripped output",
    }
    best = synth.pick(responses, task_tag="general")
    # After stripping <think>, reasoning-model has ~20 chars vs chat-model's ~120.
    # chat-model should win on length, not be drowned out by hidden reasoning.
    assert "much longer concrete answer" in best
    assert "<think>" not in best


def test_pick_code_strips_thinking_before_ast_parse():
    """Code-task picker must strip <think> so ast.parse sees actual code."""
    synth = Synthesizer()
    responses = {
        "reasoning-model": "<think>The user wants a function.</think>```python\ndef foo():\n    return 1\n```",
        "chat-model": "totally invalid python here (",
    }
    best = synth.pick(responses, task_tag="code")
    assert "def foo" in best
    assert "<think>" not in best


def test_pick_all_thinking_no_answer_treated_as_failure():
    """If a response is *only* a <think> block with no actual answer after,
    it shouldn't masquerade as a valid response."""
    synth = Synthesizer()
    responses = {
        "broken": "<think>I am still thinking...</think>",
        "good": "here is the answer",
    }
    best = synth.pick(responses, task_tag="general")
    assert best == "here is the answer"


def test_pick_code_handles_oversized_input():
    """Pathologically long candidates should not crash ast.parse."""
    synth = Synthesizer()
    huge = "x = 1\n" * 100_000  # ~600KB, exceeds the AST cap
    responses = {
        "glm": huge,
        "minimax": "def ok():\n    return 1",
    }
    best = synth.pick(responses, task_tag="code")
    # The huge input is rejected as "too big to parse"; minimax is the only
    # AST-valid candidate.
    assert best == "def ok():\n    return 1"


def test_pick_summarize_shortest():
    synth = Synthesizer()
    responses = {
        "glm": "this is a reasonably short summary",
        "minimax": "a much longer and more detailed summary that goes on and on",
    }
    best = synth.pick(responses, task_tag="summarize")
    assert best == "this is a reasonably short summary"


def test_pick_general_weak_consensus_longest_wins():
    synth = Synthesizer()
    responses = {
        "glm": "short",
        "minimax": "this is a much longer response with many more characters",
    }
    best = synth.pick(responses, task_tag="general")
    # Consensus is weak (very different strings), longest should win
    assert best == "this is a much longer response with many more characters"


def test_pick_general_tie_returns_dict():
    synth = Synthesizer()
    responses = {
        "glm": "abc",
        "minimax": "def",
    }
    best = synth.pick(responses, task_tag="general")
    # Consensus is weak and there is a tie for longest length
    assert best == {"glm": "abc", "minimax": "def"}


def test_pick_math_consensus():
    synth = Synthesizer()
    responses = {
        "glm": "The answer is 42 because 6 times 7 equals 42.",
        "minimax": "The answer is 42 because 6 * 7 = 42.",
        "deepseek": "I think the answer is 99.",
    }
    best = synth.pick(responses, task_tag="math")
    # glm and minimax are similar (consensus), so one of them should win
    assert "42" in best


def test_all_models_failed():
    synth = Synthesizer()
    responses = {
        "glm": None,
        "minimax": None,
    }
    best = synth.pick(responses, task_tag="general")
    assert best == "(all models failed)"


def test_single_valid_response():
    synth = Synthesizer()
    responses = {
        "glm": None,
        "minimax": "only valid response",
    }
    best = synth.pick(responses, task_tag="general")
    assert best == "only valid response"


def test_empty_string_responses():
    synth = Synthesizer()
    responses = {
        "glm": "",
        "minimax": "  ",
        "deepseek": "valid content here",
    }
    best = synth.pick(responses, task_tag="general")
    # Empty strings and whitespace-only should be filtered out
    assert best == "valid content here"

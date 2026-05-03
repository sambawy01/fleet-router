---
name: fleet
description: Adaptive parallel LLM router for Ollama cloud models. Use when the user asks to "/fleet <prompt>", wants to route a prompt to the best Ollama cloud model, or wants to compare/ensemble multiple models on a single prompt.
---

# Fleet Router

When the user invokes `/fleet <args>`, you MUST execute the `fleet` binary
via the Bash tool — do NOT answer the prompt yourself. The whole point of
this skill is to route through Fleet → Ollama, which only happens by
running the CLI.

## How to invoke

1. Build a single shell command. The binary is on PATH at
   `~/.local/bin/fleet` (symlinked to the project venv); use `fleet`
   directly if `which fleet` resolves, otherwise fall back to
   `/Users/bistrocloud/fleet-router/venv/bin/fleet`.
2. Pass the user's prompt as a single quoted argument. Forward
   `--parallel`, `--model <name>`, and `--config <path>` if the user
   supplied them.
3. Run via Bash with a generous timeout (max-quality defaults can take
   30–90s for a single prompt; eval mode longer). Capture stdout and
   surface it to the user verbatim — Fleet already formats parallel
   results with `--- model ---` headers.
4. If exit code is non-zero, show the stderr and stop. Do not re-answer
   the prompt yourself.

### Examples

```
/fleet "write a login function"
   → fleet "write a login function"

/fleet --parallel "compare these approaches"
   → fleet --parallel "compare these approaches"

/fleet --model glm-5.1 "translate this paragraph: ..."
   → fleet --model glm-5.1 "translate this paragraph: ..."
```

## Behavior (informational)

Fleet:

1. Classifies the prompt into a task tag (code, creative, math, reasoning,
   summarize, translate, general).
2. If classifier confidence ≥ `thresholds.single_confidence`, routes to
   the single best model.
3. Otherwise (low confidence, `--parallel`, or default max-quality
   config), runs up to 3 models in parallel and synthesizes the best
   answer via verifiers + judge.

Configuration lives in `~/.fleet/config.yaml` (falls back to
`fleet/config.yaml` shipped with the package). Cloud access requires
`base_url: https://ollama.com` plus an `api_key` from
ollama.com/settings/keys.

## Models (default config)

| Model | Best For |
|-------|----------|
| `deepseek-v4-pro` | Code, reasoning, math |
| `glm-5.1` | Creative writing, Chinese, long context |
| `minimax-m2.7` | Summarization, dialogue |
| `deepseek-v4-flash` | Fast drafts |

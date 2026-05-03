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

1. Build the command. The binary is on PATH at `~/.local/bin/fleet`;
   prefer `fleet`, fall back to
   `/Users/bistrocloud/fleet-router/venv/bin/fleet`.
2. Pass the user's prompt as a single quoted argument. Forward
   `--parallel`, `--model <name>`, and `--config <path>` if supplied.
3. Suppress sentence-transformers/HF Hub boilerplate so the user sees
   the answer, not loader noise. Pipe stderr through grep:

       fleet [flags] "<prompt>" 2> >(grep -v -E "huggingface_hub|Loading weights|Batches:|Warning: You are sending unauthenticated") 1>&1

   Or merge+filter both streams:

       fleet [flags] "<prompt>" 2>&1 | grep -v -E "huggingface_hub|Loading weights|Batches:|Warning: You are"

4. Bash `timeout`: pass `timeout: 300000` (5 minutes). Max-quality
   defaults fan out to 3 models × N samples + verifier + judge +
   optional escalation/refinement — 60–120s is normal, 180s+ on cold
   start with sentence-transformers loading. The default 2-minute
   timeout will clip many runs; do not assume the call hung.
5. If the user wants a fast answer, suggest `--model deepseek-v4-flash`
   which skips parallel synthesis and returns in ~5–10s.
6. Surface stdout verbatim — fleet already formats parallel results
   with `--- model ---` headers. If exit code is non-zero, show stderr
   and stop. Do NOT re-answer the prompt yourself; that defeats the
   skill's purpose.

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

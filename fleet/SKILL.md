---
name: fleet
description: Adaptive parallel LLM router for Ollama cloud models. Use when the user asks to "/fleet <prompt>", wants to route a prompt to the best Ollama cloud model, or wants to compare/ensemble multiple models on a single prompt.
---

# Fleet Router

Invoke with: `/fleet <prompt>`

An adaptive parallel LLM router that auto-classifies prompts and routes them to the best Ollama cloud model. Uses parallel ensemble mode when confidence is low.

## Usage

```
/fleet "write a login function"              → single model
/fleet --parallel "write a login function"   → force parallel
/fleet --model glm-5.1 "write a poem"        → override model
```

## Behavior

1. **Classify** the prompt into a task tag (code, creative, math, reasoning, summarize, translate, general)
2. If confidence is high (≥ 0.8), route to the single best model
3. If confidence is low or `--parallel` flag used, run up to 3 models in parallel and synthesize the best answer

## Models

| Model | Best For |
|-------|----------|
| `deepseek-v4-pro` | Code, reasoning, math |
| `glm-5.1` | Creative writing, Chinese, long context |
| `minimax-m2.7` | Summarization, dialogue |
| `deepseek-v4-flash` | Fast drafts |

## Configuration

Edit `~/.fleet/config.yaml` to adjust thresholds and model priorities. If absent, the router falls back to the bundled `fleet/config.yaml` shipped with the package.

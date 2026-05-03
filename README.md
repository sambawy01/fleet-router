# Fleet Router

Adaptive parallel LLM router that automatically classifies prompts and routes them to the best Ollama cloud model. Uses parallel ensemble mode when confidence is low.

## What it does

1. **Classifies** your prompt into a task tag (code, creative, math, reasoning, summarize, translate, general)
2. **Routes** high-confidence prompts to a single best-matched model for speed
3. **Runs in parallel** low-confidence prompts across multiple models and synthesizes the best answer

## Models

| Model | Best For | Ollama Tag |
|-------|----------|------------|
| DeepSeek V4 Pro | Code, reasoning, math | `deepseek-v4-pro:cloud` |
| GLM 5.1 | Creative writing, Chinese, long context | `glm-5.1:cloud` |
| MiniMax 2.7 | Summarization, dialogue | `minimax-m2.7:cloud` |
| DeepSeek V4 Flash | Fast drafts | `deepseek-v4-flash:cloud` |

## Architecture

```
Prompt → Classifier → Decision → Dispatcher → Synthesizer → Result
              ↑            ↑            ↑              ↑
         keywords +    confidence   parallel or     pick best by
         embeddings     threshold    single          task heuristic
```

## Installation

```bash
git clone https://github.com/sambawy01/fleet-router.git
cd fleet-router
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Auto-route (single or parallel based on confidence)
fleet "write a login function"

# Force parallel ensemble
fleet --parallel "write a login function"

# Force a specific model
fleet --model glm-5.1 "write a poem"

# Custom config
fleet --config ~/.fleet/config.yaml "refactor this auth module"
```

### Python

```python
import asyncio
from fleet import FleetRouter, load_config

config = load_config()
router = FleetRouter(config)

async def main():
    result = await router.ask("write a python function")
    print(result)

asyncio.run(main())
```

### Claude Code Skill

Add `fleet/SKILL.md` to your `~/.claude/skills/` directory and invoke with:

```
/fleet "write a login function"
```

## Configuration

Create `~/.fleet/config.yaml`:

```yaml
ollama:
  base_url: http://localhost:11434

models:
  deepseek-v4-pro:
    tags: [code, reasoning, math]
    priority: 1
  glm-5.1:
    tags: [creative, chinese, long_context]
    priority: 2
  minimax-m2.7:
    tags: [summarize, dialogue]
    priority: 3

thresholds:
  single_confidence: 0.8
  parallel_timeout: 60
  max_parallel: 3

classifier:
  embeddings_model: all-MiniLM-L6-v2
```

## How it works

### Task Classification

Uses keyword matching (fast) + optional sentence-transformer embeddings (accurate) to map prompts to task tags. Confidence is reduced when uncertainty markers like "best", "compare", or "review" are detected.

### Routing Decision

| Confidence | Mode | Models Called |
|-----------|------|---------------|
| ≥ 0.8 | Single | 1 (best match) |
| < 0.8 | Parallel | Up to 3 (top matches) |

### Synthesis

Rule-based selection — no extra LLM call:

- **Code**: syntax-valid via `ast.parse` → longest valid → longest overall
- **Creative**: highest lexical diversity (unique words / total words) → longest
- **Math/Reasoning**: pairwise similarity via `difflib` → longest explanation
- **Summarize**: shortest non-empty response
- **General**: consensus score (similarity to other answers) → longest → return all

## Testing

```bash
pytest tests/ -v
```

55 tests covering:
- Config loading, defaults, malformed YAML
- Keyword classification, embeddings, uncertainty penalty
- Model discovery, priority sorting, fallback
- Async dispatch, HTTP errors, timeouts, empty strings
- Syntax validation, lexical diversity, consensus logic
- End-to-end orchestration with mocked Ollama

## Requirements

- Python 3.12+
- Ollama with cloud models (or any OpenAI-compatible endpoint)
- Optional: `sentence-transformers` for embedding-based classification

## License

MIT

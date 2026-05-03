"""Provider abstraction.

This project ships only the Ollama provider — the design goal is open-source
LLMs hosted on Ollama (local or `:cloud` tags). The Provider Protocol exists
for testability (Verifier/Classifier accept any Provider) and to allow future
Ollama-compatible backends (vLLM, LM Studio, llama.cpp HTTP) without rewriting
the call sites. It is **not** a multi-vendor SDK — OpenAI/Anthropic/etc. are
intentionally not supported.
"""
from fleet.providers.base import GenerateRequest, ModelInfo, Provider
from fleet.providers.ollama import OllamaProvider
from fleet.providers.pool import ProviderPool

__all__ = [
    "GenerateRequest",
    "ModelInfo",
    "OllamaProvider",
    "Provider",
    "ProviderPool",
]

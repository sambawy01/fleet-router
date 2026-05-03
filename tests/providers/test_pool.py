from fleet.config import Config
from fleet.providers.ollama import OllamaProvider
from fleet.providers.pool import ProviderPool


def test_from_config_registers_ollama_only():
    """The shipped product is Ollama-only; no other providers are
    auto-registered. Future Ollama-compatible backends can be added via
    pool.register(...)."""
    pool = ProviderPool.from_config(Config())
    assert pool.names() == ["ollama"]
    assert isinstance(pool.get("ollama"), OllamaProvider)


def test_pool_get_unknown_returns_none():
    pool = ProviderPool()
    assert pool.get("does-not-exist") is None


def test_pool_register_overwrites():
    pool = ProviderPool()
    p1 = OllamaProvider(base_url="http://a")
    p2 = OllamaProvider(base_url="http://b")
    pool.register(p1)
    pool.register(p2)
    assert pool.get("ollama") is p2


def test_pool_register_arbitrary_provider():
    """Future Ollama-compatible backends register the same way."""
    pool = ProviderPool()

    class FakeProvider:
        name = "vllm"
        async def generate(self, req): return [None]
        async def list_models(self): return []
        async def aclose(self): pass

    pool.register(FakeProvider())
    assert "vllm" in pool.names()

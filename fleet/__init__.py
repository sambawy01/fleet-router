"""Fleet Router — adaptive parallel LLM router for open-source models on Ollama."""
from fleet.config import Config, load_config
from fleet.router import FleetRouter

__all__ = [
    "Config",
    "FleetRouter",
    "load_config",
]

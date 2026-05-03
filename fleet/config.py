"""Load fleet/config.yaml into typed dataclasses."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".fleet" / "config.yaml"


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"


@dataclass(frozen=True)
class ModelEntry:
    tags: list[str] = field(default_factory=list)
    priority: int = 99


@dataclass(frozen=True)
class ThresholdConfig:
    single_confidence: float = 0.8
    parallel_timeout: int = 60
    max_parallel: int = 3


@dataclass(frozen=True)
class ClassifierConfig:
    embeddings_model: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    models: dict[str, ModelEntry] = field(default_factory=dict)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)


def _clean_model_key(key: str) -> str:
    """Strip :cloud suffix for clean lookup."""
    return key.removesuffix(":cloud")


def load_config(path: Path | None = None) -> Config:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return Config()

    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError:
        return Config()

    ollama_raw = raw.get("ollama", {})
    ollama = OllamaConfig(base_url=ollama_raw.get("base_url", "http://localhost:11434"))

    models_raw = raw.get("models", {})
    models: dict[str, ModelEntry] = {}
    for key, val in models_raw.items():
        if not isinstance(val, dict):
            continue
        clean = _clean_model_key(key)
        models[clean] = ModelEntry(
            tags=val.get("tags", []),
            priority=val.get("priority", 99),
        )

    thresh_raw = raw.get("thresholds", {})
    thresholds = ThresholdConfig(
        single_confidence=thresh_raw.get("single_confidence", 0.8),
        parallel_timeout=thresh_raw.get("parallel_timeout", 60),
        max_parallel=thresh_raw.get("max_parallel", 3),
    )

    clf_raw = raw.get("classifier", {})
    classifier = ClassifierConfig(
        embeddings_model=clf_raw.get("embeddings_model", "all-MiniLM-L6-v2")
    )

    return Config(
        ollama=ollama,
        models=models,
        thresholds=thresholds,
        classifier=classifier,
    )

"""Configuration merging and YAML loading."""

from pathlib import Path
from typing import Any

import yaml

from domain_adaptive_retrieval.config.schema import ExperimentConfig

_SECTION_FIELDS = {
    "runtime": {"device", "require_gpu", "precision", "seed"},
    "training": {
        "epochs",
        "batch_size",
        "accumulation",
        "learning_rate",
        "warmup_ratio",
        "gradient_checkpointing",
        "resume_checkpoint",
        "save_steps",
        "max_train_pairs",
        "negative_strategy",
        "negatives_per_positive",
        "positive_threshold",
    },
    "retrieval": {
        "candidate_top_k",
        "embedding_batch_size",
        "reranker_batch_size",
        "similarity_query_batch_size",
        "similarity_doc_chunk_size",
        "max_val_queries",
        "max_test_queries",
    },
    "model": {
        "embedding_model",
        "reranker_model",
        "max_sequence_length",
        "query_prefix",
    },
    "data": {
        "cache_dir",
        "custom_dir",
        "legal_dir",
        "esci_dir",
        "legal_chunk_words",
        "legal_chunk_overlap",
        "esci_locale",
        "code_max_train",
        "code_max_val",
        "code_max_test",
        "code_max_chars",
        "code_max_query_chars",
    },
    "reporting": {"publish", "hub_private"},
}


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping, rejecting malformed or missing config files."""
    with Path(path).open(encoding="utf-8") as file:
        result = yaml.safe_load(file) or {}
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return result


def flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    """Convert user-facing grouped YAML sections into a flat validated model."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key in _SECTION_FIELDS:
            if not isinstance(value, dict):
                raise ValueError(f"Config {key} must be a YAML object")
            unexpected = set(value) - _SECTION_FIELDS[key]
            if unexpected:
                raise ValueError(f"Unknown settings in {key}: {sorted(unexpected)}")
            result.update(value)
        else:
            result[key] = value
    return result


def load_config(
    path: str | Path | None = None,
    *,
    profile: str | Path | None = None,
    domain: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Merge defaults -> file -> profile -> explicit overrides; validate exactly once."""
    merged: dict[str, Any] = {}
    if path is not None:
        merged.update(flatten_config(read_yaml(path)))
    if profile is not None:
        merged.update(flatten_config(read_yaml(profile)))
    if domain is not None:
        merged["domain"] = domain
    if overrides:
        merged.update(flatten_config(overrides))
    return ExperimentConfig.model_validate(merged)

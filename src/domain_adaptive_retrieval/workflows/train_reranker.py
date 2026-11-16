"""Orchestrate reranker training and its run manifest."""

from pathlib import Path

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.reporting.artifacts import (
    dataset_fingerprint,
    write_json,
)
from domain_adaptive_retrieval.reporting.model_card import write_model_card
from domain_adaptive_retrieval.reporting.profiler import package_versions
from domain_adaptive_retrieval.training.reranker import RerankerTrainer
from domain_adaptive_retrieval.workflows.prepare import prepare_data


def train_reranker(config: ExperimentConfig, output: str | Path) -> Path:
    """Prepare and train a domain-specific cross-encoder with frozen validation pool."""
    output_dir = Path(output)
    bundle = prepare_data(config, output_dir)
    model_path = RerankerTrainer(config).train(bundle, output_dir)
    write_json(output_dir / "environment_versions.json", package_versions())
    write_json(
        output_dir / "run_manifest.json",
        {
            "model_type": "reranker",
            "dataset_hash": dataset_fingerprint(bundle),
            "model_path": str(model_path),
            "status": "trained_not_yet_tested",
        },
    )
    write_model_card(
        model_path / "README.md",
        model_type="reranker",
        parent=config.reranker_model,
        config=config,
        bundle=bundle,
        checkpoint=str(model_path),
    )
    return model_path

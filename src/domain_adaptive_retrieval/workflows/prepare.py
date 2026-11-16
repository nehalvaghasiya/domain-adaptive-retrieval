"""Dataset preparation workflow: adapters, checks, manifest and audit files."""

from pathlib import Path

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.registry import load_domain
from domain_adaptive_retrieval.data.validation import validate_bundle
from domain_adaptive_retrieval.reporting.artifacts import (
    dataset_fingerprint,
    write_json,
)


def prepare_data(config: ExperimentConfig, output: str | Path) -> RetrievalBundle:
    """Download/load, validate and audit a domain dataset without training anything."""
    bundle = load_domain(config)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(
        destination / "dataset_audit.json",
        {**validate_bundle(bundle), "fingerprint": dataset_fingerprint(bundle)},
    )
    write_json(
        destination / "split_manifest.json",
        bundle.meta.get(
            "split_manifest",
            {qid: split for split in bundle.qrels for qid in bundle.qrels[split]},
        ),
    )
    write_json(destination / "effective_config.json", config.model_dump(mode="json"))
    return bundle

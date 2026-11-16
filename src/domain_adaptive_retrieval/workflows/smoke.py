"""Entirely offline, deterministic end-to-end integration dry run."""

from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.data.adapters.custom import load_custom
from domain_adaptive_retrieval.data.validation import validate_bundle
from domain_adaptive_retrieval.evaluation.matrix import run_matrix
from domain_adaptive_retrieval.reporting.artifacts import save_benchmark, write_json
from domain_adaptive_retrieval.reporting.comparison import write_comparison
from domain_adaptive_retrieval.workflows._smoke_embedder import HashingEmbedder
from domain_adaptive_retrieval.workflows._smoke_reranker import LexicalReranker


def run_smoke(
    output: str | Path = "outputs/smoke",
    *,
    fixtures: str | Path = "tests/fixtures",
    domains: tuple[str, ...] = (
        "finance",
        "biomedical",
        "legal",
        "ecommerce",
        "code",
        "custom",
    ),
) -> dict[str, Any]:
    """Verify six adapters' canonical contract and S0–S6, without remote resources.

    Important: this does NOT train models and its metrics are strictly synthetic.
    """
    root = Path(fixtures)
    result: dict[str, Any] = {}
    for domain in domains:
        dataset = root / f"tiny_{domain}"
        protocol = {
            "finance": "full_corpus",
            "biomedical": "full_corpus",
            "custom": "full_corpus",
            "legal": "document_scoped",
            "ecommerce": "judged_candidates",
            "code": "held_out_corpus",
        }[domain]
        bundle = load_custom(dataset, protocol=protocol)
        bundle.meta["synthetic"] = True
        audit = validate_bundle(bundle)
        matrix = run_matrix(
            bundle,
            base_embedder=HashingEmbedder(salt="base"),
            tuned_embedder=HashingEmbedder(salt="adapted"),
            base_reranker=LexicalReranker(),
            tuned_reranker=LexicalReranker(adjusted=True),
            split="test",
            top_k=5,
            device="cpu",
        )
        path = Path(output) / domain
        save_benchmark(path, bundle, matrix)
        write_comparison(
            path / "comparison.md", matrix["metrics"], protocol=protocol, capped=False
        )
        write_json(path / "dataset_integrity.json", audit)
        result[domain] = {
            "systems": sorted(matrix["metrics"]),
            "audit": audit,
            "output": str(path),
            "synthetic": True,
        }
    return result

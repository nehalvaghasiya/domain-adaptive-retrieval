"""Real, held-out S0–S6 evaluation with trained checkpoints."""

from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.registry import load_domain
from domain_adaptive_retrieval.evaluation.bootstrap import paired_bootstrap
from domain_adaptive_retrieval.evaluation.matrix import run_matrix
from domain_adaptive_retrieval.models.device import check_device
from domain_adaptive_retrieval.models.embedder import EmbeddingModel
from domain_adaptive_retrieval.models.reranker import RerankerModel
from domain_adaptive_retrieval.reporting.artifacts import save_benchmark, write_json
from domain_adaptive_retrieval.reporting.comparison import write_comparison
from domain_adaptive_retrieval.reporting.profiler import package_versions


def run_benchmark(
    config: ExperimentConfig,
    embedding_checkpoint: str | Path,
    reranker_checkpoint: str | Path,
    output: str | Path,
    *,
    bundle: RetrievalBundle | None = None,
) -> dict[str, Any]:
    """Evaluate all seven real systems; never substitute base for a missing tuned model."""
    source = bundle or load_domain(config)
    for path in (embedding_checkpoint, reranker_checkpoint):
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Missing fine-tuned model directory: {path}")
    check_device(config.device, require_gpu=config.require_gpu)
    ids = source.split_queries("test")[: config.max_test_queries or None]
    base_embed = EmbeddingModel(
        config.embedding_model,
        device=config.device,
        max_length=config.max_sequence_length,
    )
    tuned_embed = EmbeddingModel(
        embedding_checkpoint,
        device=config.device,
        max_length=config.max_sequence_length,
    )
    base_rerank = RerankerModel(
        config.reranker_model,
        device=config.device,
        max_length=config.max_sequence_length,
    )
    tuned_rerank = RerankerModel(
        reranker_checkpoint, device=config.device, max_length=config.max_sequence_length
    )
    outcome = run_matrix(
        source,
        base_embedder=base_embed,
        tuned_embedder=tuned_embed,
        base_reranker=base_rerank,
        tuned_reranker=tuned_rerank,
        split="test",
        query_ids=ids,
        top_k=config.candidate_top_k,
        device=config.device,
        query_prefix=config.query_prefix,
        embedding_batch_size=config.embedding_batch_size,
        reranker_batch_size=config.reranker_batch_size,
        doc_chunk=config.similarity_doc_chunk_size,
    )
    output_path = Path(output)
    save_benchmark(output_path, source, outcome)
    write_json(output_path / "effective_config.json", config.model_dump(mode="json"))
    write_json(output_path / "environment_versions.json", package_versions())
    write_json(
        output_path / "hardware.json",
        check_device(config.device, require_gpu=config.require_gpu),
    )
    write_comparison(
        output_path / "comparison.md",
        outcome["metrics"],
        protocol=source.protocol.value,
        capped=outcome["capped_run"],
    )
    confidence: dict[str, Any] = {}
    for left, right in (("S1", "S2"), ("S3", "S4"), ("S5", "S6")):
        for metric in ("ndcg@10", "mrr@10", "legal_character_recall@5"):
            if metric in outcome["metrics"][left]:
                confidence[f"{left}_to_{right}:{metric}"] = paired_bootstrap(
                    outcome["per_query"][left],
                    outcome["per_query"][right],
                    metric=metric,
                    seed=config.seed,
                )
    write_json(output_path / "paired_bootstrap.json", confidence)
    return outcome

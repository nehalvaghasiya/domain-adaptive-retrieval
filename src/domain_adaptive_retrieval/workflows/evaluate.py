"""Resumable base-versus-tuned test evaluation for one trained model."""

import gc
import json
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.registry import load_domain
from domain_adaptive_retrieval.evaluation.bootstrap import paired_bootstrap
from domain_adaptive_retrieval.evaluation.evaluator import evaluate_predictions
from domain_adaptive_retrieval.models.checkpoint import fingerprint_path
from domain_adaptive_retrieval.models.embedder import EmbeddingModel
from domain_adaptive_retrieval.models.reranker import RerankerModel
from domain_adaptive_retrieval.reporting.artifacts import (
    dataset_fingerprint,
    write_json,
    write_predictions,
)
from domain_adaptive_retrieval.reporting.comparison import write_comparison
from domain_adaptive_retrieval.retrieval.candidates import (
    load_candidates,
    save_candidates,
)
from domain_adaptive_retrieval.retrieval.dense import DenseRetriever
from domain_adaptive_retrieval.retrieval.rerank import rerank_candidates
from domain_adaptive_retrieval.training.progress import config_fingerprint
from domain_adaptive_retrieval.training.resume import manifest_matches


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _release_model_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_checkpoint(
    config: ExperimentConfig,
    kind: str,
    checkpoint: Path,
    output: Path,
    *,
    expected_dataset_hash: str | None = None,
) -> dict[str, dict[str, float]]:
    """Reuse completed variants and frozen candidates after an interrupted evaluation."""
    if kind not in {"embedding", "reranker"}:
        raise ValueError(f"Unknown model type: {kind}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing selected model: {checkpoint}")
    bundle = load_domain(config)
    data_hash = dataset_fingerprint(bundle)
    if expected_dataset_hash is not None and data_hash != expected_dataset_hash:
        raise ValueError(
            "Evaluation dataset changed since training; refusing to mix results"
        )
    signature = {
        "config": config_fingerprint(config),
        "dataset": data_hash,
        "checkpoint": fingerprint_path(checkpoint),
        "kind": kind,
    }
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "evaluation_progress.json"
    progress: dict[str, Any] = (
        _read_json(progress_path)
        if progress_path.exists()
        else {"signature": signature, "variants": {}}
    )
    if progress["signature"] != signature:
        raise ValueError(
            "Evaluation configuration, dataset or checkpoint changed; use a new output directory"
        )
    write_json(progress_path, progress)
    qids = bundle.split_queries("test")[: config.max_test_queries or None]
    if not qids:
        raise ValueError("Evaluation requires test judgments")
    capped = bool(bundle.meta.get("capped_run")) or len(qids) < len(
        bundle.qrels["test"]
    )
    write_json(output / "effective_config.json", config.model_dump(mode="json"))
    write_json(
        output / "dataset_audit.json",
        {
            "fingerprint": data_hash,
            "protocol": bundle.protocol.value,
            "evaluated_queries": len(qids),
            "available_test_queries": len(bundle.qrels["test"]),
            "capped_run": capped,
        },
    )

    candidates = None
    if kind == "reranker" and not all(
        label in progress["variants"] for label in ("base", "tuned")
    ):
        candidate_path = output / "frozen_candidates.json"
        if candidate_path.exists():
            candidates = load_candidates(candidate_path)
        else:
            embedder = EmbeddingModel(
                config.embedding_model,
                device=config.device,
                max_length=config.max_sequence_length,
            )
            candidates = DenseRetriever(
                embedder,
                device=config.device,
                query_prefix=config.query_prefix,
                batch_size=config.embedding_batch_size,
                query_batch=config.similarity_query_batch_size,
                doc_chunk=config.similarity_doc_chunk_size,
            ).search(bundle, qids, split="test", top_k=config.candidate_top_k)
            save_candidates(candidate_path, candidates)
            del embedder
            _release_model_memory()

    metrics: dict[str, dict[str, float]] = {}
    per_query = {}
    for label in ("base", "tuned"):
        result_path = output / f"result_{label}.json"
        predictions_path = output / f"predictions_{label}.jsonl"
        if label in progress["variants"]:
            if not manifest_matches(output, progress["variants"][label]):
                raise ValueError(
                    f"Completed {label} evaluation artifacts are missing or truncated"
                )
            result = _read_json(result_path)
        else:
            name = (
                str(checkpoint)
                if label == "tuned"
                else (
                    config.embedding_model
                    if kind == "embedding"
                    else config.reranker_model
                )
            )
            print(f"Evaluating {kind}: {label} on {len(qids)} test queries", flush=True)
            model: Any
            if kind == "embedding":
                model = EmbeddingModel(
                    name, device=config.device, max_length=config.max_sequence_length
                )
                predictions = DenseRetriever(
                    model,
                    device=config.device,
                    query_prefix=config.query_prefix,
                    batch_size=config.embedding_batch_size,
                    query_batch=config.similarity_query_batch_size,
                    doc_chunk=config.similarity_doc_chunk_size,
                ).search(bundle, qids, split="test", top_k=config.candidate_top_k)
            else:
                assert candidates is not None
                model = RerankerModel(
                    name, device=config.device, max_length=config.max_sequence_length
                )
                predictions = rerank_candidates(
                    bundle, candidates, model, batch_size=config.reranker_batch_size
                )
            scores, query_scores = evaluate_predictions(
                bundle, predictions, split="test", query_ids=qids
            )
            result = {"metrics": scores, "per_query": query_scores}
            write_predictions(predictions_path, predictions, bundle)
            write_json(result_path, result)
            progress["variants"][label] = {
                path.name: path.stat().st_size
                for path in (result_path, predictions_path)
            }
            write_json(progress_path, progress)
            del model, predictions
            _release_model_memory()
        metrics[label] = result["metrics"]
        per_query[label] = result["per_query"]
        write_json(output / "metrics.json", metrics)

    write_json(output / "per_query.json", per_query)
    confidence = {
        metric: paired_bootstrap(
            per_query["base"], per_query["tuned"], metric=metric, seed=config.seed
        )
        for metric in ("ndcg@10", "mrr@10", "legal_character_recall@5")
        if metric in metrics["base"]
    }
    write_json(output / "paired_bootstrap.json", confidence)
    write_comparison(
        output / "comparison.md", metrics, protocol=bundle.protocol.value, capped=capped
    )
    progress["complete"] = True
    write_json(progress_path, progress)
    return metrics

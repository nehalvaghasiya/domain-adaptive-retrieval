"""Seven fixed retriever/reranker systems with frozen candidate ID sets."""

from collections.abc import Sequence
from typing import Any

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.evaluation.evaluator import evaluate_predictions
from domain_adaptive_retrieval.retrieval.bm25 import BM25Retriever
from domain_adaptive_retrieval.retrieval.candidates import (
    Predictions,
    allowed_ids,
    candidate_fingerprint,
)
from domain_adaptive_retrieval.retrieval.dense import DenseRetriever
from domain_adaptive_retrieval.retrieval.rerank import rerank_candidates


def run_matrix(
    bundle: RetrievalBundle,
    *,
    base_embedder: Any,
    tuned_embedder: Any,
    base_reranker: Any,
    tuned_reranker: Any,
    split: str = "test",
    query_ids: Sequence[str] | None = None,
    top_k: int = 100,
    device: str = "cpu",
    query_prefix: str = "",
    embedding_batch_size: int = 32,
    reranker_batch_size: int = 16,
    doc_chunk: int = 8192,
) -> dict[str, Any]:
    """Evaluate S0–S6, asserting S3/S4 and S5/S6 use identical candidate pools.

    All four model objects are mandatory for a complete matrix. A fake model may
    be supplied explicitly by the *offline smoke* only, never manufactured here.
    """
    qids = list(query_ids) if query_ids is not None else bundle.split_queries(split)
    if not qids:
        raise ValueError(f"No {split} queries available for benchmark")
    sparse = BM25Retriever(bundle.corpus)
    bm25: Predictions = {
        qid: sparse.search(
            bundle.queries[qid], top_k=top_k, allowed=allowed_ids(bundle, qid, split)
        )
        for qid in qids
    }

    def dense(model: Any) -> Predictions:
        return DenseRetriever(
            model,
            device=device,
            query_prefix=query_prefix,
            batch_size=embedding_batch_size,
            doc_chunk=doc_chunk,
        ).search(bundle, qids, split=split, top_k=top_k)

    s1, s2 = dense(base_embedder), dense(tuned_embedder)
    scores: dict[str, Predictions] = {
        "S0": bm25,
        "S1": s1,
        "S2": s2,
        "S3": rerank_candidates(
            bundle, s1, base_reranker, batch_size=reranker_batch_size
        ),
        "S4": rerank_candidates(
            bundle, s1, tuned_reranker, batch_size=reranker_batch_size
        ),
        "S5": rerank_candidates(
            bundle, s2, base_reranker, batch_size=reranker_batch_size
        ),
        "S6": rerank_candidates(
            bundle, s2, tuned_reranker, batch_size=reranker_batch_size
        ),
    }

    # Reranking may change order but must preserve membership per-query.
    def member_hash(predictions: Predictions) -> dict[str, list[str]]:
        return {
            qid: sorted(did for did, _ in ranking)
            for qid, ranking in predictions.items()
        }

    for left, right in (("S1", "S3"), ("S1", "S4"), ("S2", "S5"), ("S2", "S6")):
        if member_hash(scores[left]) != member_hash(scores[right]):
            raise AssertionError(f"Candidate membership changed: {left} versus {right}")
    metrics: dict[str, dict[str, float]] = {}
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    for system, ranks in scores.items():
        metrics[system], per_query[system] = evaluate_predictions(
            bundle, ranks, split=split, query_ids=qids
        )
    return {
        "metrics": metrics,
        "per_query": per_query,
        "predictions": scores,
        "candidate_hashes": {
            "base_embedder": candidate_fingerprint(s1),
            "tuned_embedder": candidate_fingerprint(s2),
        },
        "candidate_count": {qid: len(s1[qid]) for qid in qids},
        "query_count": len(qids),
        "split": split,
        "protocol": bundle.protocol.value,
        "capped_run": bool(bundle.meta.get("capped_run", False))
        or len(qids) < len(bundle.qrels[split]),
    }

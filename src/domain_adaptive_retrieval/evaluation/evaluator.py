"""Protocol-aware per-query and aggregate evaluation of ranked IDs."""

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean
from typing import TypeAlias

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.protocol import EvaluationProtocol
from domain_adaptive_retrieval.evaluation.ecommerce import esci_metrics
from domain_adaptive_retrieval.evaluation.ranking import (
    ap_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from domain_adaptive_retrieval.evaluation.spans import span_metrics
from domain_adaptive_retrieval.retrieval.candidates import Predictions, allowed_ids

PerQuery: TypeAlias = dict[str, dict[str, float]]


def query_metrics(
    bundle: RetrievalBundle, qid: str, ranked_ids: Sequence[str], split: str
) -> dict[str, float]:
    """Return only metrics meaningful for the dataset's evaluation protocol."""
    judgments = bundle.qrels[split][qid]
    if bundle.protocol == EvaluationProtocol.JUDGED_CANDIDATES:
        return {
            **esci_metrics(ranked_ids, judgments, 10),
            "ndcg@5": ndcg_at_k(ranked_ids, judgments, 5),
            "mrr@10": mrr_at_k(ranked_ids, judgments, 10, threshold=2.0),
        }
    metrics = {
        **{f"ndcg@{k}": ndcg_at_k(ranked_ids, judgments, k) for k in (5, 10)},
        **{f"recall@{k}": recall_at_k(ranked_ids, judgments, k) for k in (10, 50, 100)},
        **{f"precision@{k}": precision_at_k(ranked_ids, judgments, k) for k in (5, 10)},
        **{f"hit@{k}": hit_at_k(ranked_ids, judgments, k) for k in (5, 10)},
        "mrr@10": mrr_at_k(ranked_ids, judgments, 10),
        "map@10": ap_at_k(ranked_ids, judgments, 10),
    }
    if bundle.protocol == EvaluationProtocol.DOCUMENT_SCOPED:
        metrics.update(span_metrics(bundle, qid, ranked_ids, 5))
    return metrics


def evaluate_predictions(
    bundle: RetrievalBundle,
    predictions: Predictions,
    *,
    split: str = "test",
    query_ids: Sequence[str] | None = None,
) -> tuple[dict[str, float], PerQuery]:
    """Reject unknown ranks/out-of-scope IDs; produce macro-averaged metrics."""
    qids = list(query_ids) if query_ids is not None else bundle.split_queries(split)
    if not qids:
        raise ValueError(f"Cannot evaluate empty {split} set")
    per_query: PerQuery = {}
    for qid in qids:
        if qid not in bundle.qrels[split]:
            raise ValueError(f"Unlabeled evaluation query {qid!r} in {split}")
        ranked = predictions.get(qid, [])
        ids = [did for did, _ in ranked]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicated candidate results for query {qid!r}")
        if not set(ids) <= set(allowed_ids(bundle, qid, split)):
            raise ValueError(
                f"Query {qid!r} includes docs outside declared candidate scope"
            )
        per_query[qid] = query_metrics(bundle, qid, ids, split)
    measures = sorted(set().union(*(set(m.keys()) for m in per_query.values())))
    aggregate = {
        key: float(fmean(per_query[q][key] for q in qids if key in per_query[q]))
        for key in measures
    }
    if bundle.protocol == EvaluationProtocol.DOCUMENT_SCOPED:
        by_source: dict[str, list[float]] = defaultdict(list)
        for qid in qids:
            by_source[qid.split(":")[0]].append(
                per_query[qid]["legal_character_recall@5"]
            )
        aggregate["legal_source_balanced_char_recall@5"] = float(
            fmean(fmean(vals) for vals in by_source.values())
        )
    return aggregate, per_query

"""Amazon ESCI judged-candidate metric policy."""

from collections.abc import Mapping, Sequence

from domain_adaptive_retrieval.evaluation.ranking import hit_at_k, mrr_at_k, ndcg_at_k


def esci_metrics(
    ranked_ids: Sequence[str], judgments: Mapping[str, float], k: int = 10
) -> dict[str, float]:
    """Graded nDCG plus exact-match MRR; never report invented full-corpus recall."""
    return {
        f"ndcg@{k}": ndcg_at_k(ranked_ids, judgments, k),
        f"exact_mrr@{k}": mrr_at_k(ranked_ids, judgments, k, threshold=2.0),
        f"exact_hit@{k}": hit_at_k(ranked_ids, judgments, k, threshold=2.0),
    }

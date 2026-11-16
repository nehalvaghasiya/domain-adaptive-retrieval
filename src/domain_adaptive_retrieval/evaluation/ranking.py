"""Pure, label-aware ranking metrics with documented denominators."""

import math
from collections.abc import Mapping, Sequence


def dcg_at_k(grades: Sequence[float], k: int) -> float:
    """Graded gain 2**relevance-1, log2(rank+1) discount."""
    return sum(
        (2 ** float(grade) - 1) / math.log2(i + 2) for i, grade in enumerate(grades[:k])
    )


def ndcg_at_k(
    ranked_ids: Sequence[str], judgments: Mapping[str, float], k: int
) -> float:
    """Judged relevance NDCG; unjudged are zero, a known sparse-qrels limitation."""
    ideal = sorted(judgments.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if not ideal_dcg:
        return 0.0
    return dcg_at_k([judgments.get(did, 0.0) for did in ranked_ids], k) / ideal_dcg


def recall_at_k(
    ranked_ids: Sequence[str],
    judgments: Mapping[str, float],
    k: int,
    threshold: float = 0.0,
) -> float:
    """Fraction of positively judged documents found, not corpus-wide true recall."""
    gold = {did for did, grade in judgments.items() if grade > threshold}
    return len(gold.intersection(ranked_ids[:k])) / len(gold) if gold else 0.0


def precision_at_k(
    ranked_ids: Sequence[str],
    judgments: Mapping[str, float],
    k: int,
    threshold: float = 0.0,
) -> float:
    """Precision with fixed denominator k; unreturned positions count as misses."""
    if k <= 0:
        raise ValueError("k must be positive")
    gold = {did for did, grade in judgments.items() if grade > threshold}
    return len(gold.intersection(ranked_ids[:k])) / k


def mrr_at_k(
    ranked_ids: Sequence[str],
    judgments: Mapping[str, float],
    k: int = 10,
    threshold: float = 0.0,
) -> float:
    """Reciprocal rank of the first result above the relevance threshold."""
    return next(
        (
            1 / i
            for i, did in enumerate(ranked_ids[:k], 1)
            if judgments.get(did, 0) > threshold
        ),
        0.0,
    )


def ap_at_k(
    ranked_ids: Sequence[str],
    judgments: Mapping[str, float],
    k: int = 10,
    threshold: float = 0.0,
) -> float:
    """Truncated average precision divided by min(total relevant, k)."""
    gold = {did for did, grade in judgments.items() if grade > threshold}
    if not gold:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, did in enumerate(ranked_ids[:k], 1):
        if did in gold:
            hits += 1
            precision_sum += hits / index
    return precision_sum / min(len(gold), k)


def hit_at_k(
    ranked_ids: Sequence[str],
    judgments: Mapping[str, float],
    k: int = 10,
    threshold: float = 0.0,
) -> float:
    """One if at least one positively judged result occurs in top-k."""
    return float(any(judgments.get(did, 0) > threshold for did in ranked_ids[:k]))

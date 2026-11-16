"""Deterministic positive and random-negative training sampling."""

import random

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.mining.exclusions import negative_ids


def positive_pairs(
    bundle: RetrievalBundle, *, threshold: float = 1.0, limit: int = 0, seed: int = 42
) -> list[tuple[str, str]]:
    """Sample only explicitly positive train labels, never dev/test judgments."""
    pairs = [
        (qid, did)
        for qid, judges in bundle.qrels["train"].items()
        for did, rel in judges.items()
        if rel >= threshold
    ]
    random.Random(seed).shuffle(pairs)
    return pairs[:limit] if limit else pairs


def random_negatives(
    bundle: RetrievalBundle,
    qid: str,
    did: str,
    *,
    n: int = 2,
    seed: int = 42,
    threshold: float = 1.0,
) -> list[str]:
    """Sample permitted weak negatives with deterministic seed."""
    possible = negative_ids(bundle, qid, did, threshold=threshold)
    return random.Random(seed).sample(possible, k=min(n, len(possible)))

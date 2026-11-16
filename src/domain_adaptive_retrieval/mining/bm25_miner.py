"""BM25 hard negatives plus randomized negatives, scoped and leakage-aware."""

import random

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.mining.exclusions import negative_ids
from domain_adaptive_retrieval.retrieval.bm25 import BM25Retriever


def mine_training_negatives(
    bundle: RetrievalBundle,
    pairs: list[tuple[str, str]],
    *,
    per_positive: int = 2,
    strategy: str = "bm25",
    threshold: float = 1.0,
    seed: int = 42,
) -> list[tuple[str, str, list[str]]]:
    """Mine relevant-looking but not known-positive samples for training only."""
    if per_positive <= 0 or strategy not in {"bm25", "random"}:
        raise ValueError("Use per_positive>0 and strategy=BM25 or random")
    rng = random.Random(seed)
    bm25 = BM25Retriever(bundle.corpus) if strategy == "bm25" else None
    result: list[tuple[str, str, list[str]]] = []
    for qid, positive in pairs:
        candidates = negative_ids(bundle, qid, positive, threshold=threshold)
        if not candidates:
            continue
        hard = []
        if bm25 and len(candidates) > 1:
            hard = [
                did
                for did, _ in bm25.search(
                    bundle.queries[qid], top_k=per_positive - 1, allowed=candidates
                )
            ]
        remainder = [did for did in candidates if did not in hard]
        chosen = hard + rng.sample(
            remainder, k=min(len(remainder), per_positive - len(hard))
        )
        if chosen:
            result.append((qid, positive, chosen))
    if not result:
        raise ValueError(
            "No training pairs with permissible negatives; enlarge corpus or add judged negatives"
        )
    return result

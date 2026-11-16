"""Cross-encoder scoring over frozen candidate lists."""

from typing import Any

import numpy as np

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.retrieval.candidates import Predictions


def rerank_candidates(
    bundle: RetrievalBundle,
    candidates: Predictions,
    model: Any,
    *,
    batch_size: int = 16,
) -> Predictions:
    """Rerank only supplied IDs without changing candidate membership."""
    results: Predictions = {}
    for qid, ranking in candidates.items():
        if not ranking:
            results[qid] = []
            continue
        ids = [did for did, _ in ranking]
        values = np.asarray(
            model.predict(
                [(bundle.queries[qid], bundle.corpus[did]) for did in ids],
                batch_size=batch_size,
                show_progress_bar=False,
            )
        ).reshape(-1)
        if len(values) != len(ids):
            raise ValueError(
                f"Reranker returned {len(values)} scores for {len(ids)} candidates"
            )
        results[qid] = sorted(
            zip(ids, map(float, values), strict=True), key=lambda row: (-row[1], row[0])
        )
    return results

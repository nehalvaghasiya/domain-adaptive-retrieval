"""Trainer-ready rows, kept separate from train/validation/test label source."""

from domain_adaptive_retrieval.data.bundle import RetrievalBundle


def embedding_triplets(
    bundle: RetrievalBundle,
    mined: list[tuple[str, str, list[str]]],
    *,
    query_prefix: str = "",
) -> list[dict[str, str]]:
    """Return one anchor, positive and hard negative per mined example."""
    return [
        {
            "anchor": query_prefix + bundle.queries[qid],
            "positive": bundle.corpus[pos],
            "negative": bundle.corpus[neg[0]],
        }
        for qid, pos, neg in mined
    ]


def reranker_pairs(
    bundle: RetrievalBundle, mined: list[tuple[str, str, list[str]]]
) -> list[dict[str, str | float]]:
    """Build pointwise BCE training data: 1 for positives, 0 for sampled negatives."""
    rows: list[dict[str, str | float]] = []
    for qid, pos, negatives in mined:
        rows.append(
            {"query": bundle.queries[qid], "document": bundle.corpus[pos], "label": 1.0}
        )
        rows.extend(
            {"query": bundle.queries[qid], "document": bundle.corpus[neg], "label": 0.0}
            for neg in negatives
        )
    return rows

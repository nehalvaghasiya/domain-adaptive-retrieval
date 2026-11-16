"""Safety rules for selecting negative training documents."""

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.protocol import EvaluationProtocol
from domain_adaptive_retrieval.retrieval.candidates import allowed_ids


def negative_ids(
    bundle: RetrievalBundle, qid: str, positive: str, *, threshold: float = 1.0
) -> list[str]:
    """Reject all known positive IDs, exact duplicate text, and out-of-scope docs.

    Non-ESCI unjudged negatives are weak labels, not verified non-relevant docs.
    """
    labels = bundle.qrels["train"].get(qid, {})
    if positive not in bundle.corpus or positive not in labels:
        raise ValueError(f"Training positive unknown: query={qid!r}, doc={positive!r}")
    relevant = {did for did, rel in labels.items() if rel >= threshold}
    candidates = allowed_ids(bundle, qid, "train")
    if bundle.train_docs is not None:
        candidates = [did for did in candidates if did in bundle.train_docs]
    return [
        did
        for did in candidates
        if did not in relevant
        and did != positive
        and bundle.corpus[did].strip() != bundle.corpus[positive].strip()
        and (bundle.protocol != EvaluationProtocol.JUDGED_CANDIDATES or did in labels)
    ]

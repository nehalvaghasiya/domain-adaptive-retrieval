"""Canonical candidate scopes and reproducible pool fingerprints."""

import hashlib
import json
from pathlib import Path
from typing import TypeAlias

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.protocol import EvaluationProtocol

Ranking: TypeAlias = list[tuple[str, float]]
Predictions: TypeAlias = dict[str, Ranking]


def allowed_ids(bundle: RetrievalBundle, query_id: str, split: str) -> list[str]:
    """Return documents permitted in this query's evaluation protocol."""
    if query_id in bundle.pools:
        return bundle.pools[query_id]
    if query_id in bundle.scopes:
        return bundle.scope_docs[bundle.scopes[query_id]]
    if bundle.protocol == EvaluationProtocol.HELD_OUT_CORPUS:
        if split not in bundle.split_docs:
            raise ValueError(
                f"Held-out corpus is not declared for split={split!r}; never infer it from relevance labels"
            )
        return bundle.split_docs[split]
    return sorted(bundle.corpus)


def candidate_fingerprint(predictions: Predictions) -> str:
    """Hash ordered candidate IDs (never reranker scores) for fairness auditing."""
    pairs = [(qid, [did for did, _ in predictions[qid]]) for qid in sorted(predictions)]
    blob = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def save_candidates(path: Path, rankings: Predictions) -> None:
    """Persist immutable candidate IDs, first-stage scores and their fingerprint."""
    from domain_adaptive_retrieval.reporting.artifacts import write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fingerprint": candidate_fingerprint(rankings), "rankings": rankings}
    write_json(path, payload)


def load_candidates(path: Path) -> Predictions:
    """Reload ranked candidates and reject a tampered or stale fingerprint."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rankings: Predictions = {
        qid: [(str(did), float(score)) for did, score in pairs]
        for qid, pairs in payload["rankings"].items()
    }
    if candidate_fingerprint(rankings) != payload["fingerprint"]:
        raise ValueError(f"Candidate pool hash mismatch: {path}")
    return rankings

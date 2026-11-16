"""Integrity checks, including the BEIR empty-document qrel regression."""

from typing import Any

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.protocol import EvaluationProtocol


class DatasetIntegrityError(ValueError):
    """Dataset violates a reference, split, or candidate-scope invariant."""


def validate_bundle(bundle: RetrievalBundle) -> dict[str, Any]:
    """Validate without silently deleting documents or fabricating qrels.

    A loader may explicitly filter *known* empty documents and attached qrels
    before this point, but genuinely absent IDs always raise an exception.
    """
    if not bundle.corpus or not bundle.queries:
        raise DatasetIntegrityError("Dataset must have non-empty queries and corpus")
    for did, content in bundle.corpus.items():
        if not str(did).strip() or not content.strip():
            raise DatasetIntegrityError(f"Empty document ID/text at document {did!r}")
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        labels = bundle.qrels.get(split, {})
        if not isinstance(labels, dict):
            raise DatasetIntegrityError(f"Expected mapping qrels[{split}]")
        for qid, docs in labels.items():
            if qid not in bundle.queries:
                raise DatasetIntegrityError(
                    f"Split {split}: qrels references missing query {qid!r}"
                )
            if qid in seen:
                raise DatasetIntegrityError(
                    f"Query {qid!r} crosses train/val/test split boundaries"
                )
            seen.add(qid)
            if not docs:
                raise DatasetIntegrityError(
                    f"Split {split}: query {qid!r} has no usable judgments"
                )
            for did, grade in docs.items():
                if did not in bundle.corpus:
                    raise DatasetIntegrityError(
                        f"Split {split}, query {qid!r}: qrels references missing corpus document {did!r}"
                    )
                if grade < 0 or grade != grade:
                    raise DatasetIntegrityError(
                        f"Split {split}, query {qid!r}, doc {did!r}: invalid grade {grade}"
                    )
        counts[split] = len(labels)
    if bundle.protocol == EvaluationProtocol.HELD_OUT_CORPUS:
        for split in ("train", "val", "test"):
            split_doc_ids = bundle.split_docs.get(split)
            if not split_doc_ids:
                raise DatasetIntegrityError(
                    f"Held-out corpus membership missing for {split!r}"
                )
            if len(split_doc_ids) != len(set(split_doc_ids)) or any(
                did not in bundle.corpus for did in split_doc_ids
            ):
                raise DatasetIntegrityError(
                    f"Invalid held-out corpus IDs for {split!r}"
                )
            for qid, query_labels in bundle.qrels[split].items():
                if not set(query_labels).issubset(split_doc_ids):
                    raise DatasetIntegrityError(
                        f"Query {qid!r} relevance doc is outside its split corpus"
                    )
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if set(bundle.split_docs[a]) & set(bundle.split_docs[b]):
                raise DatasetIntegrityError(
                    f"Held-out corpus overlaps between {a} and {b}"
                )
    for qid, pool in bundle.pools.items():
        if qid not in seen:
            raise DatasetIntegrityError(
                f"Candidate pool for unknown or unlabeled query {qid!r}"
            )
        if len(pool) != len(set(pool)):
            raise DatasetIntegrityError(
                f"Duplicate IDs in candidate pool for query {qid!r}"
            )
        split = next(s for s in ("train", "val", "test") if qid in bundle.qrels[s])
        for did in pool:
            if did not in bundle.corpus:
                raise DatasetIntegrityError(
                    f"Pool for query {qid!r} references missing doc {did!r}"
                )
            if (
                bundle.protocol == EvaluationProtocol.JUDGED_CANDIDATES
                and did not in bundle.qrels[split][qid]
            ):
                raise DatasetIntegrityError(
                    f"ESCI query {qid!r}: unjudged candidate {did!r}"
                )
    for qid, scope in bundle.scopes.items():
        if qid not in seen or not bundle.scope_docs.get(scope):
            raise DatasetIntegrityError(
                f"Query {qid!r}: missing source-document scope {scope!r}"
            )
    for did, (start, end) in bundle.offsets.items():
        if (
            did not in bundle.source_docs
            or did not in bundle.corpus
            or start < 0
            or end <= start
        ):
            raise DatasetIntegrityError(f"Malformed legal offset for document {did!r}")
    for qid, spans in bundle.gold_spans.items():
        if qid not in seen or any(start < 0 or end <= start for _, start, end in spans):
            raise DatasetIntegrityError(f"Invalid gold evidence span for query {qid!r}")
    if bundle.train_docs is not None and bundle.train_docs - bundle.corpus.keys():
        raise DatasetIntegrityError("train_docs contains unknown document IDs")
    return {
        "dataset": bundle.meta.get("dataset", "custom"),
        "protocol": bundle.protocol.value,
        "corpus_documents": len(bundle.corpus),
        "queries": len(bundle.queries),
        "split_queries": counts,
        "judgments": {k: sum(map(len, bundle.qrels[k].values())) for k in counts},
        "data_quality": bundle.meta.get("data_quality", {}),
        "capped_run": bool(bundle.meta.get("capped_run", False)),
    }

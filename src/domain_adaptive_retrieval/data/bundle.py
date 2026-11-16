"""Canonical in-memory retrieval benchmark bundle."""

from dataclasses import dataclass, field
from typing import Any

from domain_adaptive_retrieval.data.protocol import EvaluationProtocol


def _empty_qrels() -> dict[str, dict[str, dict[str, float]]]:
    return {"train": {}, "val": {}, "test": {}}


@dataclass
class RetrievalBundle:
    """Queries, corpus, labels and optional candidate/evidence scopes.

    Explicitly retains graded judgments and original legal document offsets.
    `pools` are *judged* candidate lists; `scopes` are source-document IDs.
    `split_docs` declares the permitted corpus per held-out split, independent
    of relevance judgments; never infer candidate membership from gold labels.
    """

    corpus: dict[str, str] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    qrels: dict[str, dict[str, dict[str, float]]] = field(default_factory=_empty_qrels)
    pools: dict[str, list[str]] = field(default_factory=dict)
    scopes: dict[str, str] = field(default_factory=dict)
    scope_docs: dict[str, list[str]] = field(default_factory=dict)
    source_docs: dict[str, str] = field(default_factory=dict)
    offsets: dict[str, tuple[int, int]] = field(default_factory=dict)
    gold_spans: dict[str, list[tuple[str, int, int]]] = field(default_factory=dict)
    esci_labels: dict[tuple[str, str], str] = field(default_factory=dict)
    train_docs: set[str] | None = None
    split_docs: dict[str, list[str]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def protocol(self) -> EvaluationProtocol:
        """Return a strongly typed evaluation protocol."""
        return EvaluationProtocol(self.meta.get("protocol", "full_corpus"))

    def split_queries(self, split: str) -> list[str]:
        """Return sorted query IDs with judgments for this split."""
        if split not in self.qrels:
            raise ValueError(f"Unknown split: {split}; expected train, val or test")
        return sorted(self.qrels[split])

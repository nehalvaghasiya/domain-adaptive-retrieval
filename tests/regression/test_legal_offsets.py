"""Evidence offsets and legal source grouping regressions."""

import pytest
from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.chunks import chunk_words
from domain_adaptive_retrieval.data.splits import grouped_splits
from domain_adaptive_retrieval.evaluation.spans import merge_intervals, span_metrics


def test_chunks_reproduce_exact_source_substrings() -> None:
    document = "Alpha beta\n\ngamma  delta epsilon zeta"
    chunks = list(chunk_words(document, width=3, overlap=1))
    assert chunks
    for _, start, end, text in chunks:
        assert document[start:end] == text


def test_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError):
        list(chunk_words("a b", width=2, overlap=2))


def test_grouped_source_split_no_overlap() -> None:
    a = grouped_splits([f"source{i}" for i in range(20)], seed=42)
    b = grouped_splits([f"source{i}" for i in range(20)], seed=42)
    assert a == b
    assert set(a.values()) == {"train", "val", "test"}


def test_union_no_double_counting() -> None:
    assert merge_intervals([(0, 5), (2, 7), (8, 9)]) == [(0, 7), (8, 9)]


def test_character_span_precision_recall(legal_bundle: RetrievalBundle) -> None:
    qid = legal_bundle.split_queries("test")[0]
    ranked = legal_bundle.scope_docs[legal_bundle.scopes[qid]]
    metric = span_metrics(legal_bundle, qid, ranked, k=1)
    assert metric["legal_character_recall@1"] == 1.0
    assert metric["legal_character_precision@1"] == 1.0

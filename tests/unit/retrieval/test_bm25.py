"""Lexical retrieval basic properties and stable ties."""

import pytest

from domain_adaptive_retrieval.retrieval.bm25 import BM25Retriever, tokenize


def test_tokenize_letters_and_numbers() -> None:
    assert tokenize("GPU-T4 Laptop 15GB") == ["gpu", "t4", "laptop", "15gb"]


def test_empty_corpus_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Retriever({})


def test_bm25_ranks_relevant_first() -> None:
    engine = BM25Retriever(
        {"b": "financial stocks", "a": "poetry", "c": "stock markets"}
    )
    ids = [did for did, _ in engine.search("financial stocks")]
    assert ids[0] == "b"


def test_bm25_ties_by_doc_id() -> None:
    engine = BM25Retriever({"b": "same words", "a": "same words"})
    assert [did for did, _ in engine.search("same")] == ["a", "b"]


def test_bm25_scoped_pool() -> None:
    engine = BM25Retriever(
        {"b": "financial stocks", "a": "poetry", "c": "stock markets"}
    )
    assert [did for did, _ in engine.search("stock", allowed=["a", "c"])] == ["c", "a"]

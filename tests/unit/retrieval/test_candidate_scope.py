"""Candidate IDs must stay frozen across reranker comparisons."""

from pathlib import Path

import pytest
from domain_adaptive_retrieval.retrieval.candidates import (
    candidate_fingerprint,
    load_candidates,
    save_candidates,
)


def test_candidate_fingerprint_ignores_scores() -> None:
    a = {"q": [("x", 0.4), ("y", 0.1)]}
    b = {"q": [("x", 0.9), ("y", -12.0)]}
    assert candidate_fingerprint(a) == candidate_fingerprint(b)


def test_candidate_fingerprint_is_order_sensitive() -> None:
    assert candidate_fingerprint({"q": [("x", 1), ("y", 0)]}) != candidate_fingerprint(
        {"q": [("y", 0), ("x", 1)]}
    )


def test_save_reload_candidate_hash(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    a = {"q": [("x", 0.4), ("y", 0.1)]}
    save_candidates(path, a)
    assert load_candidates(path) == a
    path.write_text(path.read_text().replace('"x"', '"z"'), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_candidates(path)

"""Paired bootstrap keeps query alignments and is seed reproducible."""

import pytest
from domain_adaptive_retrieval.evaluation.bootstrap import paired_bootstrap


def test_bootstrap_aligned_queries() -> None:
    baseline = {"q1": {"ndcg@10": 0.1}, "q2": {"ndcg@10": 0.2}}
    tuned = {"q1": {"ndcg@10": 0.6}, "q2": {"ndcg@10": 0.7}, "new": {"ndcg@10": 0.9}}
    first = paired_bootstrap(baseline, tuned, seed=42, resamples=200)
    second = paired_bootstrap(baseline, tuned, seed=42, resamples=200)
    assert first == second
    assert first["queries"] == 2
    assert first["delta"] == pytest.approx(0.5)


def test_bootstrap_missing_metric() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap({"a": {"foo": 1}}, {"a": {"foo": 2}})

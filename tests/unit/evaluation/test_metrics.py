"""Independent ranking metric denominators, empty labels and hand examples."""

import math

import pytest

from domain_adaptive_retrieval.evaluation.ranking import (
    ap_at_k,
    dcg_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_dcg_formula() -> None:
    assert dcg_at_k([3, 2, 0], 3) == pytest.approx(7 + 3 / math.log2(3))


def test_perfect_ndcg() -> None:
    assert ndcg_at_k(["best", "second"], {"best": 3, "second": 2}, 10) == pytest.approx(
        1
    )


def test_ndcg_penalizes_reversed_results() -> None:
    assert ndcg_at_k(["bad", "good"], {"good": 3, "bad": 0}, 10) < 1


def test_no_positive_metrics() -> None:
    grades = {"a": 0}
    assert ndcg_at_k(["a"], grades, 10) == 0
    assert recall_at_k(["a"], grades, 10) == 0
    assert ap_at_k(["a"], grades, 10) == 0


def test_mrr_map_recall_hit() -> None:
    labels = {"a": 1, "b": 1, "c": 0}
    ranked = ["c", "a", "b"]
    assert mrr_at_k(ranked, labels, 10) == 0.5
    assert ap_at_k(ranked, labels, 10) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert recall_at_k(ranked, labels, 2) == 0.5
    assert hit_at_k(ranked, labels, 1) == 0
    assert hit_at_k(ranked, labels, 2) == 1
    assert precision_at_k(ranked, labels, 5) == 0.4

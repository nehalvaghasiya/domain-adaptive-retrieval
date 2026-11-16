"""No ESCI corpus-wide recall; exact-label semantics and candidate enforcement."""

import pytest
from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.evaluation.ecommerce import esci_metrics
from domain_adaptive_retrieval.evaluation.evaluator import evaluate_predictions
from domain_adaptive_retrieval.retrieval.candidates import allowed_ids


def test_exact_and_substitute_grading() -> None:
    grades = {"e": 3, "s": 2, "i": 0}
    exact_first = esci_metrics(["e", "s", "i"], grades)
    substitute_first = esci_metrics(["s", "e", "i"], grades)
    assert exact_first["ndcg@10"] > substitute_first["ndcg@10"]
    assert substitute_first["exact_mrr@10"] == 0.5


def test_judged_candidate_scope(ecommerce_bundle: RetrievalBundle) -> None:
    qid = ecommerce_bundle.split_queries("test")[0]
    assert len(allowed_ids(ecommerce_bundle, qid, "test")) == 4
    _, scores = evaluate_predictions(
        ecommerce_bundle,
        {qid: [(did, 1.0) for did in allowed_ids(ecommerce_bundle, qid, "test")]},
    )
    assert "recall@100" not in scores[qid]
    with pytest.raises(ValueError, match="outside declared"):
        evaluate_predictions(ecommerce_bundle, {qid: [("ecommerce_0_0", 1.0)]})

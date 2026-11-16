"""Train examples are built without consulting held-out labels."""

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.training.examples import (
    embedding_triplets,
    reranker_pairs,
)
from domain_adaptive_retrieval.training.selection import choose_best_checkpoint


def test_training_example_conversions(finance_bundle: RetrievalBundle) -> None:
    mined = [("finance_q0", "finance_0_0", ["finance_0_1", "finance_0_2"])]
    embed = embedding_triplets(finance_bundle, mined, query_prefix="query: ")
    rank = reranker_pairs(finance_bundle, mined)
    assert embed[0]["anchor"].startswith("query: ")
    assert "positive" in embed[0] and "negative" in embed[0]
    assert [row["label"] for row in rank] == [1.0, 0.0, 0.0]


def test_best_checkpoint_val_only_ties() -> None:
    scores = {"checkpoint-20": {"ndcg@10": 0.5}, "checkpoint-10": {"ndcg@10": 0.5}}
    assert choose_best_checkpoint(scores) == "checkpoint-10"

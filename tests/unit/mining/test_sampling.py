"""No test labels, positives or out-of-scope docs in hard-negative sampling."""

import pytest

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.mining.bm25_miner import mine_training_negatives
from domain_adaptive_retrieval.mining.exclusions import negative_ids
from domain_adaptive_retrieval.mining.sampler import positive_pairs, random_negatives


def test_positive_pairs_train_only(finance_bundle: RetrievalBundle) -> None:
    assert positive_pairs(finance_bundle) == [("finance_q0", "finance_0_0")]


def test_negatives_exclude_positives(finance_bundle: RetrievalBundle) -> None:
    negatives = negative_ids(finance_bundle, "finance_q0", "finance_0_0")
    assert "finance_0_0" not in negatives
    assert len(negatives) > 0


def test_random_negatives_repeatable(finance_bundle: RetrievalBundle) -> None:
    one = random_negatives(finance_bundle, "finance_q0", "finance_0_0", seed=42)
    two = random_negatives(finance_bundle, "finance_q0", "finance_0_0", seed=42)
    assert one == two


@pytest.mark.parametrize("strategy", ["bm25", "random"])
def test_mining_works(finance_bundle: RetrievalBundle, strategy: str) -> None:
    triples = mine_training_negatives(
        finance_bundle, positive_pairs(finance_bundle), strategy=strategy
    )
    assert triples[0][0] == "finance_q0"
    assert triples[0][1] == "finance_0_0"
    assert len(triples[0][2]) > 0


def test_esci_only_judged_negatives(ecommerce_bundle: RetrievalBundle) -> None:
    negatives = negative_ids(
        ecommerce_bundle, "ecommerce_q0", "ecommerce_0_0", threshold=3
    )
    assert set(negatives) <= set(ecommerce_bundle.qrels["train"]["ecommerce_q0"])
    assert "ecommerce_0_0" not in negatives


def test_code_only_training_docs() -> None:
    from domain_adaptive_retrieval.data.adapters.custom import load_custom

    d = load_custom("tests/fixtures/tiny_code", protocol="held_out_corpus")
    d.train_docs = {did for judges in d.qrels["train"].values() for did in judges}
    # Tiny fixture contains only 4 train docs; none of the val/test ones may be mined.
    result = negative_ids(d, "code_q0", "code_0_0")
    assert all(did.startswith("code_0_") for did in result)

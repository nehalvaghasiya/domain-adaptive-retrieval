"""Negative tests for canonical data integrity."""

from copy import deepcopy

import pytest

from domain_adaptive_retrieval.data.adapters.custom import load_custom
from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.validation import DatasetIntegrityError, validate_bundle


@pytest.mark.parametrize("domain", ["finance", "biomedical", "legal", "ecommerce", "code", "custom"])
def test_every_fixture_validates(domain: str) -> None:
    from pathlib import Path

    bundle = load_custom(Path("tests/fixtures") / f"tiny_{domain}", protocol={"legal": "document_scoped", "ecommerce": "judged_candidates", "code": "held_out_corpus"}.get(domain, "full_corpus"))
    audit = validate_bundle(bundle)
    assert audit["split_queries"] == {"train": 1, "val": 1, "test": 1}


def test_missing_document_strict(finance_bundle: RetrievalBundle) -> None:
    bundle = deepcopy(finance_bundle)
    bundle.qrels["test"]["finance_q2"]["missing_doc"] = 1
    with pytest.raises(DatasetIntegrityError, match="missing corpus document 'missing_doc'"):
        validate_bundle(bundle)


def test_duplicate_query_across_splits(finance_bundle: RetrievalBundle) -> None:
    bundle = deepcopy(finance_bundle)
    bundle.qrels["val"]["finance_q2"] = {"finance_2_0": 1}
    with pytest.raises(DatasetIntegrityError, match="crosses"):
        validate_bundle(bundle)


def test_empty_document_rejected(finance_bundle: RetrievalBundle) -> None:
    bundle = deepcopy(finance_bundle)
    bundle.corpus["finance_2_0"] = ""
    with pytest.raises(DatasetIntegrityError, match="Empty document"):
        validate_bundle(bundle)


def test_judged_pool_unknown_document(ecommerce_bundle: RetrievalBundle) -> None:
    bundle = deepcopy(ecommerce_bundle)
    bundle.pools["ecommerce_q2"].append("ghost")
    with pytest.raises(DatasetIntegrityError, match="missing doc"):
        validate_bundle(bundle)


def test_esci_unjudged_candidate_rejected(ecommerce_bundle: RetrievalBundle) -> None:
    bundle = deepcopy(ecommerce_bundle)
    bundle.corpus["extra"] = "unjudged"
    bundle.pools["ecommerce_q2"].append("extra")
    with pytest.raises(DatasetIntegrityError, match="unjudged candidate"):
        validate_bundle(bundle)

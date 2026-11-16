"""Safe candidate, model card, reporting, and publishing contracts."""

import json
from pathlib import Path

import pytest
from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.reporting.artifacts import (
    dataset_fingerprint,
    write_json,
)
from domain_adaptive_retrieval.reporting.comparison import make_comparison
from domain_adaptive_retrieval.reporting.model_card import build_model_card
from domain_adaptive_retrieval.reporting.publish import publish_checkpoint


def test_json_writer_creates_parent_and_preserves_unicode(tmp_path: Path) -> None:
    file = tmp_path / "report" / "metrics.json"
    write_json(file, {"word": "café", "measure": 1.0})
    assert json.loads(file.read_text()) == {"word": "café", "measure": 1.0}


def test_dataset_fingerprint_changes_with_data(finance_bundle: RetrievalBundle) -> None:
    before = dataset_fingerprint(finance_bundle)
    finance_bundle.corpus["finance_0_0"] += " changed"
    assert dataset_fingerprint(finance_bundle) != before


def test_model_card_has_warning_and_truthful_results(
    finance_bundle: RetrievalBundle,
) -> None:
    card = build_model_card(
        model_type="embedding",
        parent="base",
        config=ExperimentConfig(),
        bundle=finance_bundle,
    )
    assert "No held-out test scores" in card
    assert "license: mit" in card
    assert "finance" in card


def test_comparison_subset_label() -> None:
    content = make_comparison(
        {"S0": {"ndcg@10": 0.5}}, protocol="full_corpus", capped=True
    )
    assert "subset/capped run" in content


def test_publish_requires_confirm_before_network(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        publish_checkpoint(tmp_path, repo_id="USER/model")


def test_publish_requires_checkpoint_card(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        publish_checkpoint(tmp_path, repo_id="USER/model", confirm=True)

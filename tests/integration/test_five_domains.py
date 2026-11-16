"""Full S0–S6 offline dry runs over every proposed domain's canonical contract."""

from pathlib import Path

import pytest

from domain_adaptive_retrieval.workflows.smoke import run_smoke


@pytest.mark.parametrize(
    "domain", ["finance", "biomedical", "legal", "ecommerce", "code", "custom"]
)
def test_domain_end_to_end(tmp_path: Path, domain: str) -> None:
    results = run_smoke(tmp_path, domains=(domain,))
    assert results[domain]["systems"] == ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    directory = tmp_path / domain
    assert (directory / "comparison.csv").is_file()
    assert (directory / "dataset_audit.json").is_file()
    assert (directory / "predictions_S6.jsonl").is_file()
    assert (directory / "candidate_hashes.json").is_file()

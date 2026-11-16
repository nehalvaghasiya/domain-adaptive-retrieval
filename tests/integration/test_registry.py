"""Route every user-facing domain name through the explicit adapter registry."""

from pathlib import Path

import pytest
from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data import registry
from domain_adaptive_retrieval.data.adapters.custom import load_custom


@pytest.mark.parametrize(
    "domain", ["finance", "biomedical", "legal", "ecommerce", "code", "custom"]
)
def test_registry_all_routes(monkeypatch: pytest.MonkeyPatch, domain: str) -> None:
    source = Path("tests/fixtures") / f"tiny_{domain}"
    reference = load_custom(
        source,
        protocol={
            "legal": "document_scoped",
            "ecommerce": "judged_candidates",
            "code": "held_out_corpus",
        }.get(domain, "full_corpus"),
    )
    monkeypatch.setattr(registry, "load_beir", lambda *_a: reference)
    monkeypatch.setattr(registry, "load_legalbench", lambda *_a, **_k: reference)
    monkeypatch.setattr(registry, "download_legalbench", lambda *_a: source)
    monkeypatch.setattr(registry, "load_esci", lambda *_a, **_k: reference)
    monkeypatch.setattr(registry, "load_codesearchnet", lambda *_a: reference)
    cfg = ExperimentConfig(
        domain=domain, custom_dir=source if domain == "custom" else None
    )
    assert registry.load_domain(cfg).corpus == reference.corpus

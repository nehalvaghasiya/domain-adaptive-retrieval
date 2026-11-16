"""Shared small canonical fixtures; no web or GPU downloads during pytest."""

from pathlib import Path

import pytest

from domain_adaptive_retrieval.data.adapters.custom import load_custom
from domain_adaptive_retrieval.data.bundle import RetrievalBundle


@pytest.fixture
def finance_bundle() -> RetrievalBundle:
    """Tiny train/dev/test fixture used by integration and metrics tests."""
    return load_custom(Path(__file__).parent / "fixtures" / "tiny_finance")


@pytest.fixture
def ecommerce_bundle() -> RetrievalBundle:
    return load_custom(Path(__file__).parent / "fixtures" / "tiny_ecommerce")


@pytest.fixture
def legal_bundle() -> RetrievalBundle:
    return load_custom(
        Path(__file__).parent / "fixtures" / "tiny_legal", protocol="document_scoped"
    )

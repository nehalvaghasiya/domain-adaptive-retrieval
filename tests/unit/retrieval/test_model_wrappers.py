"""Lazy model wrappers verify actual parameter placement, not just config strings."""

import sys
import types

import pytest
import torch
from domain_adaptive_retrieval.models.device import verify_model_device
from domain_adaptive_retrieval.models.embedder import EmbeddingModel
from domain_adaptive_retrieval.models.reranker import RerankerModel


class FakeEmbedding:
    def __init__(self, checkpoint, device):
        self.layer = torch.nn.Linear(2, 2, device=device)

    def parameters(self):
        return self.layer.parameters()

    def encode(self, texts, **options):
        return [len(text) for text in texts]


class FakeReranker:
    def __init__(self, checkpoint, device, **options):
        self.model = torch.nn.Linear(2, 2, device=device)

    def predict(self, texts, **options):
        return [0.5] * len(texts)


def test_embedding_wrapper_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeEmbedding
    module.CrossEncoder = FakeReranker
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    obj = EmbeddingModel("fake", device="cpu")
    assert obj.encode(["hello"]) == [5]
    assert obj.model.max_seq_length == 256


def test_reranker_wrapper_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeReranker
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    obj = RerankerModel("fake", device="cpu")
    assert obj.predict([("a", "b")]) == [0.5]


def test_model_location_mismatch_raises() -> None:
    with pytest.raises(RuntimeError, match="expected cuda"):
        verify_model_device(FakeEmbedding("fake", "cpu"), "cuda:0")

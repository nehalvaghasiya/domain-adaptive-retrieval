"""Mock external Transformers transport while genuinely exercising CPU orchestration.

These tests do NOT verify actual Sentence Transformers training; they exercise
our dataset conversion, checkpoint selection, validation, artifacts and reload
path without network. GPU tests run separately on real T4 hardware.
"""

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.adapters.custom import load_custom
from domain_adaptive_retrieval.training.embedder import EmbeddingTrainer
from domain_adaptive_retrieval.training.reranker import RerankerTrainer
from domain_adaptive_retrieval.workflows._smoke_embedder import HashingEmbedder
from domain_adaptive_retrieval.workflows._smoke_reranker import LexicalReranker


class FakeSentenceTransformer:
    """CUDA placement-compatible model facade for non-network orchestration tests."""

    def __init__(self, checkpoint: str, device: str = "cpu", **_: Any):
        self.checkpoint = checkpoint
        self.model = torch.nn.Linear(4, 4).to(device)
        self.max_seq_length = 256

    def parameters(self) -> Any:
        return self.model.parameters()

    def encode(self, texts: list[str], **options: Any) -> np.ndarray:
        return HashingEmbedder().encode(texts)

    def save(self, path: str) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "weights.marker").write_text("fake weights (not actual model)")


class FakeCrossEncoder:
    """Reranker facade returning lexical scores for checkpoint selection tests."""

    def __init__(self, checkpoint: str, device: str = "cpu", **_: Any):
        self.model = torch.nn.Linear(4, 4).to(device)
        self.name = checkpoint

    def predict(self, pairs: list[tuple[str, str]], **options: Any) -> np.ndarray:
        return LexicalReranker().predict(pairs)

    def save_pretrained(self, path: str) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "weights.marker").write_text("fake weights (not actual model)")


class FakeArguments:
    def __init__(self, **options: Any):
        self.__dict__.update(options)


class FakeTrainer:
    def __init__(self, model: Any, args: FakeArguments, train_dataset: Any, loss: Any):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.loss = loss

    def train(self) -> None:
        # A genuine (CPU) gradient update on the fake torch model tests orchestration.
        optimizer = torch.optim.SGD(self.model.model.parameters(), lr=0.01)
        output = self.model.model.weight.square().mean()
        output.backward()
        optimizer.step()
        (Path(self.args.output_dir) / "checkpoint-1").mkdir(parents=True)


class FakeLoss:
    def __init__(self, model: Any, **_: Any):
        self.model = model


class FakeDataset:
    @staticmethod
    def from_list(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert rows
        return rows


def install_fake_train_libs(monkeypatch: pytest.MonkeyPatch) -> None:
    datasets = types.ModuleType("datasets")
    datasets.Dataset = FakeDataset
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = FakeSentenceTransformer
    st.CrossEncoder = FakeCrossEncoder
    st.SentenceTransformerTrainer = FakeTrainer
    st.SentenceTransformerTrainingArguments = FakeArguments
    losses = types.ModuleType("sentence_transformers.losses")
    losses.MultipleNegativesRankingLoss = FakeLoss
    st.losses = losses
    training_args = types.ModuleType("sentence_transformers.training_args")
    training_args.BatchSamplers = types.SimpleNamespace(NO_DUPLICATES="no-duplicates")
    cross = types.ModuleType("sentence_transformers.cross_encoder")
    cross.CrossEncoderTrainer = FakeTrainer
    cross.CrossEncoderTrainingArguments = FakeArguments
    cross_losses = types.ModuleType("sentence_transformers.cross_encoder.losses")
    cross_losses.BinaryCrossEntropyLoss = FakeLoss
    for name, module in (
        ("datasets", datasets),
        ("sentence_transformers", st),
        ("sentence_transformers.losses", losses),
        ("sentence_transformers.training_args", training_args),
        ("sentence_transformers.cross_encoder", cross),
        ("sentence_transformers.cross_encoder.losses", cross_losses),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_embedding_training_checkpoint_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_train_libs(monkeypatch)
    bundle = load_custom("tests/fixtures/tiny_finance")
    cfg = ExperimentConfig(
        domain="finance", epochs=1, batch_size=2, accumulation=1, max_val_queries=1
    )
    checkpoint = EmbeddingTrainer(cfg).train(bundle, tmp_path / "embedding")
    assert (checkpoint / "weights.marker").is_file()
    assert (tmp_path / "embedding" / "validation_checkpoint_scores.json").is_file()
    assert (tmp_path / "embedding" / "training_stats.json").is_file()


def test_reranker_training_checkpoint_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_train_libs(monkeypatch)
    bundle = load_custom("tests/fixtures/tiny_finance")
    cfg = ExperimentConfig(
        domain="finance", epochs=1, batch_size=2, accumulation=1, max_val_queries=1
    )
    checkpoint = RerankerTrainer(cfg).train(bundle, tmp_path / "reranker")
    assert (checkpoint / "weights.marker").is_file()
    assert (tmp_path / "reranker" / "validation_candidates.json").is_file()
    assert (tmp_path / "reranker" / "training_stats.json").is_file()

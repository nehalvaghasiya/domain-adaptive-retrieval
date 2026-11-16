"""Independent package workflows produce artifacts without CLI-only logic."""

import json
from pathlib import Path

import pytest

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.adapters.custom import load_custom
from domain_adaptive_retrieval.workflows._smoke_embedder import HashingEmbedder
from domain_adaptive_retrieval.workflows._smoke_reranker import LexicalReranker


def test_prepare_writes_full_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from domain_adaptive_retrieval.workflows import prepare

    bundle = load_custom("tests/fixtures/tiny_finance")
    monkeypatch.setattr(prepare, "load_domain", lambda cfg: bundle)
    output = tmp_path / "prepared"
    assert prepare.prepare_data(ExperimentConfig(), output) is bundle
    report = json.loads((output / "dataset_audit.json").read_text())
    assert report["fingerprint"]
    assert (output / "split_manifest.json").exists()


def test_benchmark_complete_artifact_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from domain_adaptive_retrieval.workflows import benchmark

    bundle = load_custom("tests/fixtures/tiny_finance")
    monkeypatch.setattr(benchmark, "load_domain", lambda cfg: bundle)
    monkeypatch.setattr(
        benchmark,
        "EmbeddingModel",
        lambda checkpoint, **kwargs: HashingEmbedder(salt=str(checkpoint)),
    )
    monkeypatch.setattr(
        benchmark,
        "RerankerModel",
        lambda checkpoint, **kwargs: LexicalReranker(
            adjusted=str(checkpoint).endswith("tuned")
        ),
    )
    embed = tmp_path / "embed-tuned"
    rerank = tmp_path / "rerank-tuned"
    embed.mkdir()
    rerank.mkdir()
    outcome = benchmark.run_benchmark(
        ExperimentConfig(), embed, rerank, tmp_path / "results"
    )
    assert len(outcome["metrics"]) == 7
    for name in (
        "metrics.json",
        "comparison.md",
        "environment_versions.json",
        "hardware.json",
        "paired_bootstrap.json",
        "run_manifest.json",
    ):
        assert (tmp_path / "results" / name).exists()


def test_train_workflows_create_model_cards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from domain_adaptive_retrieval.workflows import train_embedder, train_reranker

    bundle = load_custom("tests/fixtures/tiny_finance")

    def prepare(cfg, dest):
        return bundle

    def fake_train(self, loaded, dest):
        path = Path(dest) / "model"
        path.mkdir(parents=True)
        return path

    monkeypatch.setattr(train_embedder, "prepare_data", prepare)
    monkeypatch.setattr(train_reranker, "prepare_data", prepare)
    monkeypatch.setattr(train_embedder.EmbeddingTrainer, "train", fake_train)
    monkeypatch.setattr(train_reranker.RerankerTrainer, "train", fake_train)
    cfg = ExperimentConfig()
    a = train_embedder.train_embedding(cfg, tmp_path / "embedding")
    b = train_reranker.train_reranker(cfg, tmp_path / "reranker")
    assert (a / "README.md").is_file()
    assert (b / "README.md").is_file()
    assert (tmp_path / "reranker" / "run_manifest.json").exists()

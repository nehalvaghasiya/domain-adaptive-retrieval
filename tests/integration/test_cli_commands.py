"""CLI acceptance tests for every documented command, without external downloads."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from domain_adaptive_retrieval.cli import app
from domain_adaptive_retrieval.data.adapters.custom import load_custom
from domain_adaptive_retrieval.workflows._smoke_embedder import HashingEmbedder
from domain_adaptive_retrieval.workflows._smoke_reranker import LexicalReranker

runner = CliRunner()


def test_cli_data_prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from domain_adaptive_retrieval.workflows import prepare

    monkeypatch.setattr(
        prepare, "load_domain", lambda cfg: load_custom("tests/fixtures/tiny_finance")
    )
    result = runner.invoke(
        app, ["data", "prepare", "--domain", "finance", "--output", str(tmp_path)]
    )
    assert result.exit_code == 0, result.exception
    assert (tmp_path / "finance" / "dataset_audit.json").exists()


def test_cli_data_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    from domain_adaptive_retrieval.data import registry

    monkeypatch.setattr(
        registry, "load_domain", lambda cfg: load_custom("tests/fixtures/tiny_finance")
    )
    result = runner.invoke(app, ["data", "audit", "--domain", "finance"])
    assert result.exit_code == 0, result.exception
    assert '"corpus_documents": 12' in result.output


def test_cli_train_entrypoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from domain_adaptive_retrieval.workflows import train_embedder, train_reranker

    monkeypatch.setattr(
        train_embedder, "train_embedding", lambda cfg, output: tmp_path / "embed"
    )
    monkeypatch.setattr(
        train_reranker, "train_reranker", lambda cfg, output: tmp_path / "rerank"
    )
    for kind in ("embedder", "reranker"):
        result = runner.invoke(
            app, ["train", kind, "--domain", "finance", "--output", str(tmp_path)]
        )
        assert result.exit_code == 0, result.exception
        assert "checkpoint:" in result.output


def test_cli_evaluate_entrypoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from domain_adaptive_retrieval.data import registry
    from domain_adaptive_retrieval.models import embedder, reranker

    monkeypatch.setattr(
        registry, "load_domain", lambda cfg: load_custom("tests/fixtures/tiny_finance")
    )
    monkeypatch.setattr(
        embedder, "EmbeddingModel", lambda name, **options: HashingEmbedder()
    )
    monkeypatch.setattr(
        reranker, "RerankerModel", lambda name, **options: LexicalReranker()
    )
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    for kind in ("embedder", "reranker"):
        result = runner.invoke(
            app,
            [
                "evaluate",
                kind,
                "--domain",
                "finance",
                "--checkpoint",
                str(checkpoint),
                "--profile",
                "cpu",
                "--output",
                str(tmp_path / kind),
            ],
        )
        assert result.exit_code == 0, result.exception
        assert (tmp_path / kind / "metrics.json").exists()


def test_cli_benchmark_entrypoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from domain_adaptive_retrieval.workflows import benchmark

    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda *args, **kwargs: {"metrics": {"S0": {"ndcg@10": 0.1}}},
    )
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--domain",
            "finance",
            "--embedder",
            str(tmp_path),
            "--reranker",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert "ndcg@10" in result.output


def test_cli_publish_entrypoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from domain_adaptive_retrieval.reporting import publish

    monkeypatch.setattr(
        publish,
        "publish_checkpoint",
        lambda *args, **kwargs: "https://huggingface.co/USER/model",
    )
    result = runner.invoke(
        app,
        [
            "publish",
            "model",
            "--run",
            str(tmp_path),
            "--repo-id",
            "USER/model",
            "--confirm",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert "huggingface.co" in result.output


def test_cli_smoke_rejects_online_mode() -> None:
    result = runner.invoke(app, ["smoke", "--no-offline"])
    assert result.exit_code != 0

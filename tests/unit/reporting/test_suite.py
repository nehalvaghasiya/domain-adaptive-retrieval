"""Imported evidence must retain labels, provenance and evaluation scope."""

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from domain_adaptive_retrieval.cli import app
from domain_adaptive_retrieval.models.checkpoint import fingerprint_path
from domain_adaptive_retrieval.reporting.evidence import (
    EvaluationRun,
    load_run,
    scan_runs,
    sha256_file,
)
from domain_adaptive_retrieval.reporting.publication import (
    card_metadata,
    evaluated_model_card,
    metric_table,
)
from domain_adaptive_retrieval.reporting.suite import (
    END,
    LEGACY_END,
    LEGACY_START,
    START,
    build_suite,
    package_model,
    update_readme,
)
from domain_adaptive_retrieval.retrieval.candidates import save_candidates


def put(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def experiment(root: Path, *, legacy: bool = False, kind: str = "embedding") -> Path:
    run = root / "finance" / kind
    model = run / "model"
    model.mkdir(parents=True)
    (model / "README.md").write_text("Original card; included in checkpoint digest.\n")
    put(model / "config.json", {"hidden_size": 3, "architectures": ["BertModel"]})
    header = json.dumps(
        {
            "weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        }
    ).encode()
    (model / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00" * 24
    )
    config = {
        "domain": "finance",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "base_embedding": "BAAI/bge-small-en-v1.5",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "max_sequence_length": 256,
        "embedding_max_length": 384,
        "query_prefix": "prefix: ",
        "epochs": 2,
        "seed": 42,
        "learning_rate": 2e-5,
        "batch_size": 4,
        "accumulation": 8,
        "precision": "fp16",
        "max_val_queries": 100,
    }
    put(run / ("config.json" if legacy else "effective_config.json"), config)
    put(
        run / ("versions.json" if legacy else "environment_versions.json"),
        {"sentence-transformers": "5.7.0"},
    )
    put(
        run / "training_stats.json",
        {
            "training_pairs": 6,
            "mined_triplets": 6,
            "training_seconds": 1.5,
            "selected_checkpoint": "/original/trainer/checkpoint-10",
            "objective": "BinaryCrossEntropyLoss"
            if kind == "reranker"
            else "MultipleNegativesRankingLoss",
        },
    )
    audit = {
        "dataset": "BEIR/fiqa",
        "protocol": "full_corpus",
        "fingerprint": "data",
        "corpus_documents": 2,
        "split_queries": {"train": 3, "val": 2, "test": 2},
    }
    put(run / "dataset_audit.json", audit)
    base, tuned = (
        ("S1_base_embedding", "S2_tuned_embedding") if legacy else ("base", "tuned")
    )
    per_query = {
        base: {
            "q1": {"ndcg@10": 0.6, "mrr@10": 0.75, "recall@100": 0.5},
            "q2": {"ndcg@10": 0.4, "mrr@10": 0.25, "recall@100": 0.5},
        },
        tuned: {
            "q1": {"ndcg@10": 0.4, "mrr@10": 0.5, "recall@100": 0.75},
            "q2": {"ndcg@10": 0.4, "mrr@10": 0.25, "recall@100": 0.75},
        },
    }
    metrics = {
        base: {"ndcg@10": 0.5, "mrr@10": 0.5, "recall@100": 0.5},
        tuned: {"ndcg@10": 0.4, "mrr@10": 0.375, "recall@100": 0.75},
    }
    eval_dir = run / "evaluation"
    put(eval_dir / "metrics.json", metrics)
    put(
        eval_dir / ("per_query_metrics.json" if legacy else "per_query.json"), per_query
    )
    put(
        eval_dir / ("bootstrap_delta.json" if legacy else "paired_bootstrap.json"),
        {
            "ndcg@10": {
                "delta": -0.1,
                "queries": 2,
                "resamples": 1000,
                "paired_bootstrap_95ci": [-0.2, 0.0],
            },
        },
    )
    for variant in (base, tuned):
        rows = []
        for qid in ("q1", "q2"):
            for rank, did in enumerate(("d1", "d2"), 1):
                rows.append(
                    json.dumps(
                        {
                            "query_id": qid,
                            "doc_id" if legacy else "document_id": did,
                            "rank": rank,
                            "score": 1.0 / rank,
                        }
                    )
                )
        (eval_dir / f"predictions_{variant}.jsonl").write_text("\n".join(rows) + "\n")
    if kind == "reranker":
        save_candidates(
            eval_dir / "frozen_candidates.json",
            {qid: [("d1", 1.0), ("d2", 0.5)] for qid in ("q1", "q2")},
        )
    if not legacy:
        put(eval_dir / "effective_config.json", config)
        put(
            eval_dir / "dataset_audit.json",
            {
                "fingerprint": "data",
                "evaluated_queries": 2,
                "capped_run": False,
            },
        )
        variants = {}
        for variant in (base, tuned):
            put(
                eval_dir / f"result_{variant}.json",
                {"metrics": metrics[variant], "per_query": per_query[variant]},
            )
            variants[variant] = {
                path.name: path.stat().st_size
                for path in (
                    eval_dir / f"result_{variant}.json",
                    eval_dir / f"predictions_{variant}.jsonl",
                )
            }
        put(
            eval_dir / "evaluation_progress.json",
            {
                "complete": True,
                "variants": variants,
                "signature": {
                    "kind": kind,
                    "dataset": "data",
                    "config": hashlib.sha256(
                        json.dumps(config, sort_keys=True).encode()
                    ).hexdigest(),
                    "checkpoint": fingerprint_path(model),
                },
            },
        )
        put(run / "run_manifest.json", {"status": "trained_not_yet_tested"})
    return run


@pytest.mark.parametrize("legacy", [False, True])
def test_import_preserves_negative_delta_and_recall_gain(
    tmp_path: Path, legacy: bool
) -> None:
    directory = experiment(tmp_path / "outputs", legacy=legacy)
    run = load_run(directory, "example")
    assert run.base["ndcg@10"] == 0.5
    assert run.tuned["ndcg@10"] == 0.4
    assert run.tuned["recall@100"] == 0.75
    assert run.bootstrap["ndcg@10"]["delta"] == -0.1
    assert run.query_count == 2
    assert run.model["parameters"] == 6
    assert all(len(file.sha256) == 64 for file in run.evidence)
    assert "-0.1000" in metric_table(run)
    assert "0.4000" in evaluated_model_card(run)
    assert bool(run.checkpoint_fingerprint) is not legacy


@pytest.mark.parametrize(
    "change", ["metric", "queries", "bootstrap", "predictions", "weights", "config"]
)
def test_conflicting_evidence_is_rejected(tmp_path: Path, change: str) -> None:
    directory = experiment(tmp_path / "outputs")
    if change == "metric":
        path = directory / "evaluation/metrics.json"
        data = json.loads(path.read_text())
        data["tuned"]["ndcg@10"] = 0.9
        put(path, data)
    elif change == "queries":
        path = directory / "evaluation/per_query.json"
        data = json.loads(path.read_text())
        data["tuned"]["q3"] = data["tuned"].pop("q2")
        put(path, data)
    elif change == "bootstrap":
        path = directory / "evaluation/paired_bootstrap.json"
        data = json.loads(path.read_text())
        data["ndcg@10"]["delta"] = 0.1
        put(path, data)
    elif change == "predictions":
        path = directory / "evaluation/predictions_tuned.jsonl"
        content = path.read_text()
        path.write_text(content.replace('"rank": 2', '"rank": 1'))
        # Keep the recorded byte count valid to exercise rank checks.
        assert len(path.read_text()) == len(content)
    elif change == "config":
        path = directory / "effective_config.json"
        data = json.loads(path.read_text())
        data["max_sequence_length"] = 512
        put(path, data)
    else:
        with (directory / "model/model.safetensors").open("ab") as stream:
            stream.write(b"x")
    with pytest.raises(ValueError):
        load_run(directory, "example")
    assert scan_runs(tmp_path / "outputs", "example")[0].status == "invalid"


def test_reranker_requires_exact_frozen_candidates(tmp_path: Path) -> None:
    directory = experiment(tmp_path / "outputs", kind="reranker")
    assert load_run(directory, "example").status == "evaluated"
    save_candidates(
        directory / "evaluation/frozen_candidates.json",
        {"q1": [("d1", 1.0)], "q2": [("d1", 1.0)]},
    )
    with pytest.raises(ValueError, match="Frozen candidate"):
        load_run(directory, "example")


def test_missing_and_synthetic_slots_are_not_scores(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    experiment(root)
    experiment(root / "smoke")
    put(
        root / "legal/reranker/job_state.json",
        {"status": "failed", "error": "Missing corpus"},
    )
    runs = scan_runs(root, "example")
    assert len(runs) == 10
    assert sum(run.status == "evaluated" for run in runs) == 1
    assert next(run for run in runs if run.id == "legal/reranker").status == "failed"
    with pytest.raises(ValueError, match="namespace"):
        scan_runs(root, "../outside")


def test_model_card_metadata_scores_and_local_example(tmp_path: Path) -> None:
    run = load_run(experiment(tmp_path / "outputs"), "example")
    metadata = card_metadata(run)
    assert metadata["base_model"] == "BAAI/bge-small-en-v1.5"
    results = metadata["model-index"][0]["results"][0]
    assert results["metrics"][0]["value"] == 0.4
    assert results["dataset"]["type"] == "BeIR/fiqa"
    card = evaluated_model_card(run, packaged=True)
    assert yaml.safe_load(card.split("---")[1]) == metadata
    assert 'SentenceTransformer(".", device="cpu")' in card
    assert 'query_prefix = "prefix: "' in card
    assert "/original/" not in card
    assert "evaluation/run.json" in card
    with pytest.raises(ValueError):
        evaluated_model_card(
            EvaluationRun(
                id="legal/reranker",
                domain="legal",
                model_type="reranker",
                source="outputs/legal/reranker",
                repo_id="example/legal",
                status="failed",
            )
        )
    run.domain = "code"
    assert (
        card_metadata(run)["model-index"][0]["results"][0]["dataset"]["config"]
        == "python"
    )
    run.domain = "ecommerce"
    assert "datasets" not in card_metadata(run)
    assert "model-index" not in card_metadata(run)


def test_readme_update_is_repeatable_and_preserves_other_edits(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("# Project\n\nUser introduction.\n\n## Setup\n\nUser setup.\n")
    update_readme(path, "Model links\n")
    first = path.read_text()
    update_readme(path, "Model links\n")
    assert path.read_text() == first
    assert first.count(START) == first.count(END) == 1
    assert "User introduction." in first and "User setup." in first
    path.write_text(START)
    with pytest.raises(ValueError, match="markers"):
        update_readme(path, "results")


def test_readme_migrates_legacy_scores_to_model_links(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text(
        f"# Project\n\n{LEGACY_START}\nOld scores: 0.1234\n{LEGACY_END}\n"
        "\n## Setup\n\nUser setup.\n"
    )
    section = "## Model\n\n[Model card](card.md)"
    update_readme(path, section)
    first = path.read_text()
    update_readme(path, section)
    assert path.read_text() == first
    assert LEGACY_START not in first and "0.1234" not in first
    assert section in first and "User setup." in first


def test_suite_generates_reports_and_packages_without_mutating_source(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    root = tmp_path / "outputs"
    directory = experiment(root, legacy=True)
    experiment(root, kind="reranker")
    evaluation = directory / "evaluation"
    for name in ("metrics.json", "per_query_metrics.json"):
        data = json.loads((evaluation / name).read_text())
        data["S0_BM25"] = data["S1_base_embedding"]
        put(evaluation / name, data)
    (evaluation / "predictions_S0_BM25.jsonl").write_text(
        (evaluation / "predictions_S1_base_embedding.jsonl").read_text()
    )
    before = fingerprint_path(root)
    output = tmp_path / "docs/results"
    (output / "assets").mkdir(parents=True)
    for name in ("legal.md", "assets/legal.svg", "assets/legal.png"):
        (output / name).write_text("Old generated report")
    (output / "notes.md").write_text("User notes")
    readme = tmp_path / "README.md"
    readme.write_text("# Project\n\nExisting description.\n")
    kwargs = {
        "root": root,
        "output": tmp_path / "docs/results",
        "cards_output": tmp_path / "docs/cards",
        "package_output": tmp_path / "dist/hf",
        "namespace": "example",
        "readme": readme,
        "featured_model": "finance/embedding",
    }
    build_suite(**kwargs)
    public = json.loads((tmp_path / "docs/results/results.json").read_text())
    assert public["featured_model"] == "finance/embedding"
    assert [run["id"] for run in public["runs"]] == ["finance/embedding"]
    assert set(public["runs"][0]["systems"]) == {
        "S1_base_embedding",
        "S2_tuned_embedding",
    }
    assert public["runs"][0]["systems"]["S2_tuned_embedding"]["ndcg@10"] == 0.4
    assert "S0_BM25" not in (output / "metrics.csv").read_text()
    assert "reranker" not in (output / "README.md").read_text()
    assert "pipeline comparisons" not in (output / "finance.md").read_text()
    assert not (output / "legal.md").exists()
    assert not (output / "assets/legal.svg").exists()
    assert not (output / "assets/legal.png").exists()
    assert (output / "notes.md").read_text() == "User notes"
    assert (
        "[Evaluation details](#evaluation)"
        in (tmp_path / "docs/cards/finance-reranker.md").read_text()
    )
    assert (tmp_path / "dist/hf/dar-finance-reranker-v1/model.safetensors").is_file()
    assert "0.4000" not in readme.read_text()
    assert "nDCG" not in readme.read_text()
    package = tmp_path / "dist/hf/dar-finance-embedding-v1"
    assert sha256_file(package / "model.safetensors") == sha256_file(
        directory / "model/model.safetensors"
    )
    assert "0.4000" in (package / "README.md").read_text()
    assert not (package / "trainer").exists()
    manifest = json.loads((package / "release_manifest.json").read_text())
    for name, value in manifest["files"].items():
        assert sha256_file(package / name) == value["sha256"]
    assert (tmp_path / "docs/results/assets/finance.svg").is_file()
    assert "../cards/" in (tmp_path / "docs/results/finance.md").read_text()
    snapshot = (
        readme.read_text(),
        (tmp_path / "docs/results/results.json").read_text(),
    )
    build_suite(**kwargs)
    assert snapshot == (
        readme.read_text(),
        (tmp_path / "docs/results/results.json").read_text(),
    )
    assert fingerprint_path(root) == before


def test_destinations_and_existing_packages_are_protected(tmp_path: Path) -> None:
    directory = experiment(tmp_path / "outputs")
    with pytest.raises(ValueError, match="separate"):
        build_suite(
            root=tmp_path / "outputs",
            output=directory / "docs",
            cards_output=tmp_path / "cards",
        )
    with pytest.raises(ValueError, match="separate"):
        build_suite(
            root=tmp_path / "outputs",
            output=tmp_path / "docs",
            cards_output=tmp_path / "cards",
            package_output=tmp_path / "docs/hf",
        )
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError, match="unrelated"):
        package_model(directory, target, load_run(directory, "example"))


@pytest.mark.parametrize("featured", ["legal/embedding", "unknown/model"])
def test_unavailable_featured_model_does_not_change_docs(
    tmp_path: Path, featured: str
) -> None:
    root = tmp_path / "outputs"
    experiment(root)
    output = tmp_path / "docs/results"
    output.mkdir(parents=True)
    index = output / "README.md"
    index.write_text("Existing evaluation")
    with pytest.raises(ValueError, match="no validated evaluation"):
        build_suite(
            root=root,
            output=output,
            cards_output=tmp_path / "docs/cards",
            featured_model=featured,
        )
    assert index.read_text() == "Existing evaluation"


def test_cli_suite_is_discoverable_and_rejects_source_overwrite(tmp_path: Path) -> None:
    runner = CliRunner()
    assert "--hf-namespace" in runner.invoke(app, ["report", "suite", "--help"]).stdout
    assert (
        "--featured-model" in runner.invoke(app, ["report", "suite", "--help"]).stdout
    )
    root = tmp_path / "outputs"
    root.mkdir()
    result = runner.invoke(
        app,
        [
            "report",
            "suite",
            "--input",
            str(root),
            "--output",
            str(root / "docs"),
            "--cards-output",
            str(tmp_path / "cards"),
        ],
    )
    assert result.exit_code != 0
    assert "separate" in result.output

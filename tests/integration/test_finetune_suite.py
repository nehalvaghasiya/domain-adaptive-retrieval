"""Resume failed stages and skip completed jobs without any real model downloads."""

import json
from pathlib import Path

import pytest

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.adapters.custom import load_custom
from domain_adaptive_retrieval.reporting.artifacts import (
    dataset_fingerprint,
    write_json,
)
from domain_adaptive_retrieval.training.progress import config_fingerprint
from domain_adaptive_retrieval.workflows import evaluate
from domain_adaptive_retrieval.workflows import finetune_suite as suite
from domain_adaptive_retrieval.workflows._smoke_embedder import HashingEmbedder
from domain_adaptive_retrieval.workflows._smoke_reranker import LexicalReranker


def read(path: Path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("kind", ["embedding", "reranker"])
def test_evaluation_resumes_after_tuned_variant_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    bundle = load_custom("tests/fixtures/tiny_finance")
    monkeypatch.setattr(evaluate, "load_domain", lambda cfg: bundle)
    config = ExperimentConfig()
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"unit test stand-in")
    calls = []
    fail = {"active": True}

    def embedding(name, **kwargs):
        calls.append(("embedding", name))
        if kind == "embedding" and name == str(checkpoint) and fail["active"]:
            raise RuntimeError("interrupted tuned evaluation")
        return HashingEmbedder()

    def reranker(name, **kwargs):
        calls.append(("reranker", name))
        if name == str(checkpoint) and fail["active"]:
            raise RuntimeError("interrupted tuned evaluation")
        return LexicalReranker()

    monkeypatch.setattr(evaluate, "EmbeddingModel", embedding)
    monkeypatch.setattr(evaluate, "RerankerModel", reranker)
    output = tmp_path / "evaluation"
    with pytest.raises(RuntimeError, match="interrupted tuned evaluation"):
        evaluate.evaluate_checkpoint(config, kind, checkpoint, output)
    assert "base" in read(output / "evaluation_progress.json")["variants"]
    fail["active"] = False
    metrics = evaluate.evaluate_checkpoint(config, kind, checkpoint, output)
    assert set(metrics) == {"base", "tuned"}
    assert calls.count(("embedding", config.embedding_model)) == 1
    if kind == "reranker":
        assert calls.count(("reranker", config.reranker_model)) == 1
        assert (output / "frozen_candidates.json").is_file()
    count = len(calls)
    assert evaluate.evaluate_checkpoint(config, kind, checkpoint, output) == metrics
    assert len(calls) == count
    with pytest.raises(ValueError, match="configuration"):
        evaluate.evaluate_checkpoint(
            config.model_copy(update={"candidate_top_k": 50}), kind, checkpoint, output
        )
    with pytest.raises(ValueError, match="dataset changed since training"):
        evaluate.evaluate_checkpoint(
            config, kind, checkpoint, output, expected_dataset_hash="different"
        )


def fake_results(config, kind, checkpoint, output, **kwargs):
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "metrics.json",
        "per_query.json",
        "paired_bootstrap.json",
        "evaluation_progress.json",
        "result_base.json",
        "result_tuned.json",
        "dataset_audit.json",
        "effective_config.json",
    ):
        write_json(
            output / name,
            {"base": {"ndcg@10": 0.1}, "tuned": {"ndcg@10": 0.2}}
            if name == "metrics.json"
            else {},
        )
    for name in ("comparison.md", "predictions_base.jsonl", "predictions_tuned.jsonl"):
        (output / name).write_text("unit test artifacts\n")
    if kind == "reranker":
        write_json(output / "frozen_candidates.json", {})
    return read(output / "metrics.json")


def test_seven_job_suite_continues_and_skips_completed_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle_hash = dataset_fingerprint(load_custom("tests/fixtures/tiny_finance"))
    trained = []
    attempts = []
    fail_first = {"active": True}

    def fake_train(config, output):
        trained.append(output.relative_to(tmp_path).as_posix())
        if (
            config.domain == "code"
            and output.name == "embedding"
            and fail_first["active"]
        ):
            raise RuntimeError("recoverable training failure")
        model = output / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"fake model")
        write_json(output / "run_manifest.json", {"dataset_hash": bundle_hash})
        return model

    def worker(key, output):
        attempts.append(key)
        return suite.run_job(output / key, key.split("/")[1])

    monkeypatch.setattr(suite, "preflight", lambda config: {"test_only": True})
    monkeypatch.setattr(suite, "_run_worker", worker)
    monkeypatch.setattr(suite, "train_embedding", fake_train)
    monkeypatch.setattr(suite, "train_reranker", fake_train)
    monkeypatch.setattr(suite, "evaluate_checkpoint", fake_results)
    profile = Path("configs/experiments/local_gpu.yaml")
    assert suite.run_suite(tmp_path, profile, list(suite.DEFAULT_JOBS)) == 1
    records = read(tmp_path / "finetuning_summary.json")["jobs"]
    assert records["code/embedding"]["status"] == "failed"
    assert sum(record["status"] == "complete" for record in records.values()) == 6
    fail_first["active"] = False
    assert suite.run_suite(tmp_path, profile, list(suite.DEFAULT_JOBS)) == 0
    assert attempts == [*suite.DEFAULT_JOBS, "code/embedding"]
    for key in suite.DEFAULT_JOBS:
        assert trained.count(key) == (2 if key == "code/embedding" else 1)
    assert (
        "ecommerce,embedding,complete,tuned,ndcg@10,0.2"
        in (tmp_path / "finetuning_summary.csv").read_text()
    )
    assert (
        sum(
            read(tmp_path / "finetuning_summary.json")["jobs"][key]["status"]
            == "complete"
            for key in suite.DEFAULT_JOBS
        )
        == 7
    )


def test_evaluation_failure_keeps_trained_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = ExperimentConfig()
    write_json(tmp_path / "job_config.json", config.model_dump(mode="json"))
    write_json(
        tmp_path / "job_state.json",
        {
            "config_fingerprint": config_fingerprint(config),
            "status": "pending",
            "stage": "training",
        },
    )
    calls = []

    def train(config, output):
        calls.append("train")
        model = output / "model"
        model.mkdir()
        (model / "weights.bin").write_bytes(b"unit test weights")
        write_json(output / "run_manifest.json", {"dataset_hash": "unit-test-data"})
        return model

    def fail_evaluation(*args, **kwargs):
        raise RuntimeError("test evaluation failed")

    monkeypatch.setattr(suite, "train_reranker", train)
    monkeypatch.setattr(suite, "evaluate_checkpoint", fail_evaluation)
    assert suite.run_job(tmp_path, "reranker") == 1
    assert read(tmp_path / "job_state.json")["trained"]
    monkeypatch.setattr(suite, "evaluate_checkpoint", fake_results)
    assert suite.run_job(tmp_path, "reranker") == 0
    assert calls == ["train"]


def test_dry_run_has_no_downloads_or_output_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("dry run reached GPU/network work")

    monkeypatch.setattr(suite, "preflight", unexpected)
    monkeypatch.setattr(suite, "_run_worker", unexpected)
    output = tmp_path / "outputs"
    assert suite.main(["--dry-run", "--output-dir", str(output)]) == 0
    printed = capsys.readouterr().out
    for key in suite.DEFAULT_JOBS:
        assert key in printed
    assert not output.exists()


def test_changed_settings_refused_before_worker_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key = "code/embedding"
    write_json(
        tmp_path / key / "job_state.json",
        {"config_fingerprint": "different", "status": "failed"},
    )
    with pytest.raises(ValueError, match="Configuration changed"):
        suite.run_suite(tmp_path, Path("configs/experiments/local_gpu.yaml"), [key])


def test_parallel_suite_cannot_write_locked_outputs(tmp_path: Path) -> None:
    from filelock import FileLock, Timeout

    with FileLock(tmp_path / ".finetuning.lock"):
        with pytest.raises(Timeout):
            suite.run_suite(
                tmp_path, Path("configs/experiments/local_gpu.yaml"), ["code/embedding"]
            )
    assert not (tmp_path / "code" / "embedding" / "job_state.json").exists()


def test_dataset_identity_includes_legal_evidence_and_training_scope() -> None:
    bundle = load_custom("tests/fixtures/tiny_legal", protocol="document_scoped")
    original = dataset_fingerprint(bundle)
    qid = next(iter(bundle.gold_spans))
    source, start, end = bundle.gold_spans[qid][0]
    bundle.gold_spans[qid][0] = (source, start + 1, end)
    changed = dataset_fingerprint(bundle)
    assert changed != original
    bundle.train_docs = {next(iter(bundle.corpus))}
    assert dataset_fingerprint(bundle) != changed


def test_real_worker_process_finishes_cached_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from domain_adaptive_retrieval.training.resume import file_manifest

    config = ExperimentConfig(
        domain="custom", custom_dir=Path("examples/custom_dataset")
    )
    key = "code/embedding"
    job_dir = tmp_path / key
    model = job_dir / "model"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"unit test; never loaded")
    bundle = load_custom("examples/custom_dataset")
    monkeypatch.setattr(evaluate, "load_domain", lambda cfg: bundle)
    monkeypatch.setattr(
        evaluate, "EmbeddingModel", lambda *args, **kwargs: HashingEmbedder()
    )
    evaluate.evaluate_checkpoint(config, "embedding", model, job_dir / "evaluation")
    write_json(job_dir / "job_config.json", config.model_dump(mode="json"))
    write_json(
        job_dir / "job_state.json",
        {
            "job": key,
            "config_fingerprint": config_fingerprint(config),
            "status": "running",
            "stage": "evaluation",
            "trained": {
                "files": file_manifest(model),
                "dataset_hash": dataset_fingerprint(bundle),
            },
        },
    )
    assert suite._run_worker(key, tmp_path) == 0
    state = read(job_dir / "job_state.json")
    assert suite._finished(job_dir, state)
    assert "Completed custom/embedding" in (job_dir / "run.log").read_text()

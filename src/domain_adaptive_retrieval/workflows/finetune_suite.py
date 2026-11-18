"""Run the requested seven fine-tuning jobs with persistent, independent progress."""

import argparse
import csv
import importlib.metadata
import io
import json
import os
import signal
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.config.loader import load_config
from domain_adaptive_retrieval.config.presets import REPO_ROOT, domain_config
from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.models.device import check_device
from domain_adaptive_retrieval.reporting.artifacts import write_json, write_text
from domain_adaptive_retrieval.training.progress import config_fingerprint
from domain_adaptive_retrieval.training.resume import (
    file_manifest,
    latest_resume_checkpoint,
    manifest_matches,
)
from domain_adaptive_retrieval.workflows.evaluate import evaluate_checkpoint
from domain_adaptive_retrieval.workflows.train_embedder import train_embedding
from domain_adaptive_retrieval.workflows.train_reranker import train_reranker

DEFAULT_JOBS = (
    "code/embedding",
    "ecommerce/embedding",
    "finance/reranker",
    "biomedical/reranker",
    "legal/reranker",
    "ecommerce/reranker",
    "code/reranker",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _finished(job_dir: Path, state: dict[str, Any]) -> bool:
    return (
        state.get("status") == "complete"
        and manifest_matches(job_dir / "model", state["trained"]["files"])
        and manifest_matches(job_dir, state["result_files"])
    )


def preflight(config: ExperimentConfig) -> dict[str, Any]:
    """Check dependencies and real CUDA allocation before loading public datasets."""
    versions = {}
    for name in (
        "torch",
        "sentence-transformers",
        "transformers",
        "datasets",
        "accelerate",
        "pyarrow",
        "filelock",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "Install dependencies: uv sync --extra train --extra benchmark --extra cuda126 --group dev"
            ) from exc
    if not config.require_gpu or not config.device.startswith("cuda:"):
        raise ValueError("This runner requires a CUDA profile with require_gpu=true")
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or os.environ.get("LOCAL_RANK", "-1") != "-1"
    ):
        raise ValueError("Run this script directly on one GPU, without torchrun/DDP")
    try:
        hardware = check_device(config.device, require_gpu=True)
    except RuntimeError as exc:
        import torch

        raise RuntimeError(
            f"GPU preflight failed (Torch {torch.__version__}, CUDA build {torch.version.cuda}): {exc}. See the README's local GPU setup."
        ) from exc
    return {"hardware": hardware, "versions": versions, "checked_at": _now()}


def run_job(job_dir: Path, kind: str) -> int:
    """Worker process: finish training, then resume independent test evaluation."""
    state_path = job_dir / "job_state.json"
    state = _read(state_path)
    try:
        config = ExperimentConfig.model_validate_json(
            (job_dir / "job_config.json").read_text(encoding="utf-8")
        )
        if state["config_fingerprint"] != config_fingerprint(config):
            raise ValueError("Job configuration changed; use a new output directory")
        state.update(status="running", updated_at=_now(), error=None)
        if not state.get("trained"):
            state["stage"] = "training"
            write_json(state_path, state)
            train = train_embedding if kind == "embedding" else train_reranker
            model_path = train(config, job_dir)
            files = file_manifest(model_path)
            if not files:
                raise FileNotFoundError("Training did not save a selected model")
            state["trained"] = {
                "files": files,
                "dataset_hash": _read(job_dir / "run_manifest.json")["dataset_hash"],
            }
            write_json(state_path, state)
        elif not manifest_matches(job_dir / "model", state["trained"]["files"]):
            raise ValueError("Previously trained model files are missing or truncated")
        state["stage"] = "evaluation"
        write_json(state_path, state)
        evaluate_checkpoint(
            config,
            kind,
            job_dir / "model",
            job_dir / "evaluation",
            expected_dataset_hash=state["trained"]["dataset_hash"],
        )
        result_files = [
            "metrics.json",
            "per_query.json",
            "paired_bootstrap.json",
            "comparison.md",
            "evaluation_progress.json",
            "result_base.json",
            "result_tuned.json",
            "predictions_base.jsonl",
            "predictions_tuned.jsonl",
            "dataset_audit.json",
            "effective_config.json",
        ]
        if kind == "reranker":
            result_files.append("frozen_candidates.json")
        state["result_files"] = {
            f"evaluation/{name}": (job_dir / "evaluation" / name).stat().st_size
            for name in result_files
        }
        state.update(status="complete", stage="complete", updated_at=_now(), error=None)
        write_json(state_path, state)
        print(f"Completed {config.domain}/{kind}: {job_dir}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        state.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            updated_at=_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        write_json(state_path, state)
        traceback.print_exc()
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


def _run_worker(key: str, output: Path) -> int:
    """Isolate CUDA memory and errors to one job, and tee all output to its log."""
    log_path = output / key / "run.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        "domain_adaptive_retrieval.workflows.finetune_suite",
        "--worker-job",
        key,
        "--output-dir",
        str(output),
    ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nStarted attempt at {_now()}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()


def write_summary(output: Path) -> None:
    """Record per-job status and per-domain metrics without combining domain scores."""
    jobs = {}
    text = [
        "# Fine-tuning results",
        "",
        "| Job | Status | Stage | Results |",
        "| :--- | :--- | :--- | :--- |",
    ]
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["domain", "model_type", "status", "variant", "metric", "value"])
    for key in DEFAULT_JOBS:
        job_dir = output / key
        state_path = job_dir / "job_state.json"
        state = (
            _read(state_path)
            if state_path.exists()
            else {"status": "not_started", "stage": "pending"}
        )
        metrics_path = job_dir / "evaluation" / "metrics.json"
        metrics = _read(metrics_path) if metrics_path.exists() else {}
        jobs[key] = {**state, "metrics": metrics, "output": str(job_dir)}
        comparison = (
            f"[comparison]({key}/evaluation/comparison.md)"
            if (job_dir / "evaluation/comparison.md").is_file()
            else "Pending"
        )
        text.append(f"| {key} | {state['status']} | {state['stage']} | {comparison} |")
        domain, kind = key.split("/")
        for variant, scores in metrics.items():
            for metric, value in scores.items():
                writer.writerow([domain, kind, state["status"], variant, metric, value])
        if not metrics:
            writer.writerow([domain, kind, state["status"], "", "", ""])
    write_json(output / "finetuning_summary.json", {"updated_at": _now(), "jobs": jobs})
    write_text(output / "finetuning_summary.csv", csv_buffer.getvalue())
    write_text(
        output / "finetuning_summary.md",
        "\n".join(text)
        + "\n\nDetailed base/tuned scores: `<domain>/<model_type>/evaluation/comparison.md`.\n",
    )


def run_suite(
    output: Path,
    profile: Path,
    jobs: list[str],
    *,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> int:
    """Skip completed jobs; continue failed/interrupted jobs with unchanged settings."""
    configs = {
        key: load_config(
            domain_config(key.split("/")[0]), profile=profile, domain=key.split("/")[0]
        )
        for key in jobs
    }
    for key, config in configs.items():
        if config.resume_checkpoint is not None:
            raise ValueError(
                "Leave resume_checkpoint unset; this runner discovers each job's checkpoint"
            )
        state_path = output / key / "job_state.json"
        state = _read(state_path) if state_path.exists() else {}
        if state and state["config_fingerprint"] != config_fingerprint(config):
            raise ValueError(f"Configuration changed for {key}; use a new --output-dir")
        checkpoint = latest_resume_checkpoint(output / key / "trainer")
        print(
            f"{key}: {state.get('status', 'not_started')}; resume={checkpoint or 'no saved optimizer checkpoint'}; batch={config.batch_size}; epochs={config.epochs}; save_steps={config.save_steps}",
            flush=True,
        )
    if dry_run:
        print(
            f"Output: {output}\nCode defaults: at most 15000 train / 500 validation / 500 test examples (profile may override). Other domains use their adapters' full data. No training started."
        )
        return 0
    from filelock import FileLock

    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".finetuning.lock", timeout=0):
        pending = []
        for key, config in configs.items():
            job_dir = output / key
            state_path = job_dir / "job_state.json"
            if state_path.exists() and _read(state_path)[
                "config_fingerprint"
            ] != config_fingerprint(config):
                raise ValueError(
                    f"Configuration changed for {key}; use a new --output-dir"
                )
            if state_path.exists() and _finished(job_dir, _read(state_path)):
                print(f"Skipping completed job: {key}", flush=True)
                continue
            if not state_path.exists():
                write_json(
                    state_path,
                    {
                        "job": key,
                        "config_fingerprint": config_fingerprint(config),
                        "status": "pending",
                        "stage": "training",
                        "created_at": _now(),
                    },
                )
            write_json(job_dir / "job_config.json", config.model_dump(mode="json"))
            pending.append(key)
        write_summary(output)
        if not pending:
            return 0
        try:
            write_json(output / "gpu_preflight.json", preflight(configs[pending[0]]))
        except Exception as exc:
            write_json(
                output / "preflight_error.json",
                {"error": str(exc), "checked_at": _now()},
            )
            raise
        failures = 0
        for key in pending:
            try:
                code = _run_worker(key, output)
            except KeyboardInterrupt:
                code = 130
            state_path = output / key / "job_state.json"
            state = _read(state_path)
            if code or not _finished(output / key, state):
                failures += 1
                if state["status"] not in {"failed", "interrupted"}:
                    state.update(
                        status="interrupted" if code == 130 else "failed",
                        updated_at=_now(),
                        error=f"Worker exited with code {code}; see run.log",
                    )
                    write_json(state_path, state)
            write_summary(output)
            if code == 130:
                return 130
            if failures and fail_fast:
                return 1
        print(f"Summary: {output / 'finetuning_summary.md'}", flush=True)
        return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resume two embedding jobs and five reranker jobs; rerun the same command after a failure."
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--profile", type=Path, default=REPO_ROOT / "configs/experiments/local_gpu.yaml"
    )
    parser.add_argument(
        "--jobs", nargs="+", choices=DEFAULT_JOBS, default=list(DEFAULT_JOBS)
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show jobs and resume points without requiring a GPU or downloading data",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed job instead of trying the others",
    )
    parser.add_argument("--worker-job", choices=DEFAULT_JOBS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    output = args.output_dir.resolve()
    if args.worker_job:
        return run_job(output / args.worker_job, args.worker_job.split("/")[1])
    try:
        return run_suite(
            output,
            args.profile.resolve(),
            list(dict.fromkeys(args.jobs)),
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Prevent accidental optimizer-state resets and cross-run resume mistakes."""

import json
from pathlib import Path

import pytest

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.training.resume import (
    latest_resume_checkpoint,
    mark_checkpoint_complete,
    verified_resume_checkpoint,
)


def test_no_resume_is_safe(tmp_path: Path) -> None:
    assert verified_resume_checkpoint(None, tmp_path) is None


def test_resume_requires_same_run_and_optimizer(tmp_path: Path) -> None:
    runner = tmp_path / "run" / "trainer"
    checkpoint = runner / "checkpoint-120"
    checkpoint.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="trainer_state"):
        verified_resume_checkpoint(checkpoint, runner)
    (checkpoint / "trainer_state.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="optimizer"):
        verified_resume_checkpoint(checkpoint, runner)
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-placeholder")
    assert verified_resume_checkpoint(checkpoint, runner) == str(checkpoint.resolve())


def test_resume_rejects_external_paths(tmp_path: Path) -> None:
    outsider = tmp_path / "other-run" / "checkpoint-1"
    outsider.mkdir(parents=True)
    with pytest.raises(ValueError, match="inside trainer"):
        verified_resume_checkpoint(outsider, tmp_path / "run" / "trainer")


def test_resume_config_and_cuda_device() -> None:
    cfg = ExperimentConfig(
        device="cuda",
        precision="fp16",
        resume_checkpoint=Path("outputs/my-run/trainer/checkpoint-1"),
    )
    assert cfg.device == "cuda:0"
    assert cfg.resume_checkpoint is not None


def make_complete_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        (checkpoint / name).write_bytes(b"test checkpoint contents")
    mark_checkpoint_complete(checkpoint)
    return checkpoint


def test_latest_checkpoint_ignores_interrupted_save(tmp_path: Path) -> None:
    earlier = make_complete_checkpoint(tmp_path, 100)
    latest = make_complete_checkpoint(tmp_path, 200)
    partial = tmp_path / "checkpoint-300"
    partial.mkdir()
    (partial / "trainer_state.json").write_text('{"global_step": 300}')
    assert latest_resume_checkpoint(tmp_path) == latest
    (latest / "optimizer.pt").write_bytes(b"truncated")
    assert latest_resume_checkpoint(tmp_path) == earlier


def test_checkpoint_requires_model_scheduler_and_rng(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 1}')
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    with pytest.raises(FileNotFoundError):
        mark_checkpoint_complete(checkpoint)
    assert latest_resume_checkpoint(tmp_path) is None


def test_checkpoint_rejects_invalid_state_and_corrupt_marker(tmp_path: Path) -> None:
    checkpoint = make_complete_checkpoint(tmp_path, 10)
    (checkpoint / "checkpoint_complete.json").write_text("{partial json")
    assert latest_resume_checkpoint(tmp_path) is None
    (checkpoint / "trainer_state.json").write_text('{"global_step": 11}')
    with pytest.raises(ValueError, match="step"):
        mark_checkpoint_complete(checkpoint)

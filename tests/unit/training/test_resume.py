"""Prevent accidental optimizer-state resets and cross-run resume mistakes."""

from pathlib import Path

import pytest
from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.training.resume import verified_resume_checkpoint


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

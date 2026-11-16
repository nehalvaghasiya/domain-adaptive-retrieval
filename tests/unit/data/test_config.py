"""Settings and CUDA normalization regression."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from domain_adaptive_retrieval.config.loader import flatten_config, load_config
from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.models.device import check_device, normalize_device


@pytest.mark.parametrize("value,expected", [("cuda", "cuda:0"), ("cuda:0", "cuda:0"), ("CPU", "cpu")])
def test_cuda_normalization(value: str, expected: str) -> None:
    assert normalize_device(value) == expected


def test_config_can_validate_cuda0_without_gpu() -> None:
    cfg = ExperimentConfig(device="cuda", require_gpu=True, precision="fp16")
    assert cfg.device == "cuda:0"


def test_cpu_profile_is_not_fp16() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(device="cpu", precision="fp16")


def test_require_gpu_rejects_cpu() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(require_gpu=True, device="cpu")


def test_validation_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(ignored=True)


def test_nested_unknown_key_explained() -> None:
    with pytest.raises(ValueError, match="Unknown settings"):
        flatten_config({"training": {"epohcs": 3}})


def test_config_profile_overrides(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    profile = tmp_path / "profile.yaml"
    base.write_text("domain: finance\ntraining:\n  batch_size: 4\n", encoding="utf-8")
    profile.write_text("training:\n  batch_size: 8\n", encoding="utf-8")
    cfg = load_config(base, profile=profile, overrides={"training": {"batch_size": 12}})
    assert cfg.batch_size == 12


def test_cpu_hardware() -> None:
    assert check_device("cpu")["device"] == "cpu"


def test_unavailable_cuda_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        check_device("cuda:0", require_gpu=True)

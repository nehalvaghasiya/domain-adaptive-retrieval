"""Publishing opt-in and timing guards."""

import sys
import types
from pathlib import Path

import pytest
from domain_adaptive_retrieval.models.checkpoint import fingerprint_path, model_size_mb
from domain_adaptive_retrieval.reporting.profiler import package_versions, time_call
from domain_adaptive_retrieval.reporting.publish import publish_checkpoint


def test_checkpoint_hash_stable(tmp_path: Path) -> None:
    (tmp_path / "weights.bin").write_bytes(b"hello" * 500)
    first = fingerprint_path(tmp_path)
    assert first == fingerprint_path(tmp_path)
    assert model_size_mb(tmp_path) > 0
    (tmp_path / "weights.bin").write_bytes(b"changed")
    assert fingerprint_path(tmp_path) != first


def test_checkpoint_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fingerprint_path(tmp_path / "missing")


def test_environment_versions() -> None:
    values = package_versions()
    assert "python" in values and "torch" in values


def test_time_call_cpu() -> None:
    value, timing = time_call(lambda: 42)
    assert value == 42
    assert timing["device"] == "cpu"
    assert timing["seconds"] >= 0


def test_publish_calls_hub_only_when_confirmed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "README.md").write_text("# Tested model card")
    calls: list[dict] = []

    class FakeHfApi:
        def __init__(self, token=None):
            calls.append({"token_provided": token is not None})

        def create_repo(self, repo_id, repo_type, private, exist_ok):
            calls.append({"repo": repo_id, "private": private})

        def upload_folder(self, repo_id, folder_path, repo_type):
            calls.append({"upload": folder_path})
            return "https://huggingface.co/USER/model"

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    value = publish_checkpoint(
        tmp_path, repo_id="USER/model", token="not-logged", confirm=True
    )
    assert value.endswith("USER/model")
    assert calls[1]["private"] is True
    assert "not-logged" not in str(calls)


def test_invalid_repo_id_rejected(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Card")
    with pytest.raises(ValueError, match="NAMESPACE"):
        publish_checkpoint(tmp_path, repo_id="missing-namespace", confirm=True)

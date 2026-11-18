"""The legal entrypoint collects one job's results without touching documentation."""

import runpy
from pathlib import Path

import pytest

from domain_adaptive_retrieval.workflows import finetune_suite

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_legal_reranker.py"


def test_legal_script_dry_run_has_no_downloads_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("dry run reached GPU or training work")

    monkeypatch.setattr(finetune_suite, "preflight", unexpected)
    monkeypatch.setattr(finetune_suite, "_run_worker", unexpected)
    output = tmp_path / "results"
    main = runpy.run_path(str(SCRIPT))["main"]
    assert main(["--dry-run", "--output-dir", str(output)]) == 0
    printed = capsys.readouterr().out
    assert "legal/reranker:" in printed
    assert "finance/reranker:" not in printed
    assert not output.exists()


def test_legal_script_forwards_only_legal_job_and_returns_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(finetune_suite, "main", lambda args: calls.append(args) or 1)
    main = runpy.run_path(str(SCRIPT))["main"]
    output, profile = tmp_path / "outputs", tmp_path / "profile.yaml"
    assert main(["--output-dir", str(output), "--profile", str(profile)]) == 1
    assert calls == [
        [
            "--jobs",
            "legal/reranker",
            "--output-dir",
            str(output),
            "--profile",
            str(profile),
        ]
    ]

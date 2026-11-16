"""Entry points documented in README must remain runnable."""

import json
from pathlib import Path

from typer.testing import CliRunner

from domain_adaptive_retrieval.cli import app

RUNNER = CliRunner()


def test_cli_help() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "smoke" in result.output


def test_hardware_cpu() -> None:
    result = RUNNER.invoke(app, ["hardware", "check", "--device", "cpu"])
    assert result.exit_code == 0
    assert "cpu" in result.output


def test_config_validation() -> None:
    result = RUNNER.invoke(
        app, ["config", "validate", "--config", "configs/domains/finance.yaml"]
    )
    assert result.exit_code == 0
    assert '"domain": "finance"' in result.output


def test_cli_offline_smoke(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "smoke",
            "--offline",
            "--config",
            "configs/experiments/smoke.yaml",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert len(json.loads(result.output)) == 6
    assert (tmp_path / "finance" / "metrics.json").is_file()


def test_report_from_saved_smoke(tmp_path: Path) -> None:
    result = RUNNER.invoke(app, ["smoke", "--output", str(tmp_path)])
    assert result.exit_code == 0
    report = RUNNER.invoke(app, ["report", "build", "--run", str(tmp_path / "finance")])
    assert report.exit_code == 0, report.exception
    assert (tmp_path / "finance" / "comparison.md").exists()

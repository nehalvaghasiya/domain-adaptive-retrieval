"""Persist training phases so validation failures never repeat completed training."""

import hashlib
import json
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.reporting.artifacts import (
    dataset_fingerprint,
    write_json,
)
from domain_adaptive_retrieval.training.resume import file_manifest, manifest_matches


def config_fingerprint(config: ExperimentConfig) -> str:
    values = config.model_dump(mode="json", exclude={"resume_checkpoint"})
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class TrainingProgress:
    """Track optimization, checkpoint validation and selected-model saving."""

    def __init__(
        self, config: ExperimentConfig, bundle: RetrievalBundle, target: Path, kind: str
    ) -> None:
        self.target = target
        self.path = target / "training_progress.json"
        signature = {
            "config": config_fingerprint(config),
            "dataset": dataset_fingerprint(bundle),
            "kind": kind,
        }
        if self.path.exists():
            self.state: dict[str, Any] = json.loads(
                self.path.read_text(encoding="utf-8")
            )
            if self.state.get("signature") != signature:
                raise ValueError(
                    "Training configuration or dataset changed; use a new output directory"
                )
        else:
            if (target / "model").exists() or (target / "training_stats.json").exists():
                raise ValueError(
                    "Existing model has no matching progress record; use a new output directory"
                )
            self.state = {"signature": signature, "validation_scores": {}}
            write_json(self.path, self.state)

    @property
    def optimized(self) -> bool:
        record = self.state.get("optimization")
        if record and not manifest_matches(
            self.target / "last_checkpoint", record["files"]
        ):
            raise ValueError("Completed training model files are missing or truncated")
        return bool(record)

    @property
    def selected(self) -> bool:
        record = self.state.get("selection")
        if record and not manifest_matches(self.target / "model", record["files"]):
            raise ValueError("Selected model files are missing or truncated")
        return bool(record) and (self.target / "training_stats.json").is_file()

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self.state["optimization"]["stats"])

    @property
    def scores(self) -> dict[str, dict[str, float]]:
        return {
            path: {key: float(value) for key, value in metrics.items()}
            for path, metrics in self.state["validation_scores"].items()
        }

    def finish_optimization(self, stats: dict[str, Any]) -> None:
        files = file_manifest(self.target / "last_checkpoint")
        if not files:
            raise FileNotFoundError("Training returned without saving the final model")
        self.state["optimization"] = {"stats": stats, "files": files}
        write_json(self.path, self.state)

    def record_score(self, checkpoint: Path, scores: dict[str, float]) -> None:
        self.state["validation_scores"][str(checkpoint)] = scores
        write_json(self.path, self.state)
        write_json(self.target / "validation_checkpoint_scores.json", self.scores)

    def finish_selection(self, selected: str) -> None:
        files = file_manifest(self.target / "model")
        if not files:
            raise FileNotFoundError("Checkpoint selection did not save a model")
        self.state["selection"] = {"checkpoint": selected, "files": files}
        write_json(self.path, self.state)

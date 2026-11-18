"""Validate resumable Transformers optimizer state, not just model weights."""

import json
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.reporting.artifacts import write_json
from domain_adaptive_retrieval.training.selection import checkpoints


def file_manifest(root: Path) -> dict[str, int]:
    """Record saved file sizes after a model/checkpoint save has finished."""
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"README.md", "checkpoint_complete.json"}
    }


def manifest_matches(root: Path, manifest: dict[str, Any]) -> bool:
    """Reject missing/truncated files and malformed manifest paths."""
    if not isinstance(manifest, dict) or not manifest:
        return False
    try:
        return all(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and (root / name).is_file()
            and (root / name).stat().st_size == size
            for name, size in manifest.items()
        )
    except OSError:
        return False


def mark_checkpoint_complete(checkpoint: Path) -> None:
    """Mark a complete single-GPU Trainer save, including optimizer and RNG state."""
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint / name}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    step = int(checkpoint.name.rsplit("-", 1)[-1])
    if state.get("global_step") != step:
        raise ValueError(f"Checkpoint step does not match Trainer state: {checkpoint}")
    files = file_manifest(checkpoint)
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise FileNotFoundError(f"Checkpoint has no model weights: {checkpoint}")
    write_json(checkpoint / "checkpoint_complete.json", {"step": step, "files": files})


def latest_resume_checkpoint(trainer_root: Path) -> Path | None:
    """Find the newest fully saved checkpoint, ignoring interrupted saves."""
    for checkpoint in reversed(checkpoints(trainer_root)):
        try:
            marker = json.loads(
                (checkpoint / "checkpoint_complete.json").read_text(encoding="utf-8")
            )
            state = json.loads(
                (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
            )
            if marker["step"] == state["global_step"] == int(
                checkpoint.name.rsplit("-", 1)[-1]
            ) and manifest_matches(checkpoint, marker["files"]):
                return checkpoint.resolve()
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def verified_resume_checkpoint(
    requested: Path | None, trainer_root: Path
) -> str | None:
    """Permit resuming only from a checkpoint belonging to this run.

    An arbitrary model folder may contain weights but not optimizer state, so it
    cannot safely continue the same training trajectory. A resume checkpoint
    must be located under the expected trainer output path and contain both
    trainer and optimizer state.
    """
    if requested is None:
        return None
    root = trainer_root.resolve()
    checkpoint = Path(requested).resolve()
    if not checkpoint.is_relative_to(root):
        raise ValueError(f"Resume checkpoint must be inside trainer directory: {root}")
    for filename in ("trainer_state.json", "optimizer.pt"):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(
                f"Unsafe/incomplete resume checkpoint: {checkpoint / filename}"
            )
    return str(checkpoint)

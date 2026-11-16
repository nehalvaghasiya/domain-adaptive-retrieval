"""Validate resumable Transformers optimizer state, not just model weights."""

from pathlib import Path


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

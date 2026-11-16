"""Validation-only checkpoint selection with stable tie policy."""

from collections.abc import Mapping
from pathlib import Path


def choose_best_checkpoint(
    scores: Mapping[str, Mapping[str, float]], *, metric: str = "ndcg@10"
) -> str:
    """Maximize held-out validation score, tie-break lexicographically by checkpoint path."""
    eligible = [
        (value[metric], path) for path, value in scores.items() if metric in value
    ]
    if not eligible:
        raise ValueError(f"No validation checkpoints contain metric={metric!r}")
    highest = max(score for score, _ in eligible)
    return sorted(path for score, path in eligible if score == highest)[0]


def checkpoints(root: Path) -> list[Path]:
    """List numeric Hugging Face Trainer checkpoints in chronological order."""
    return sorted(
        (
            p
            for p in root.glob("checkpoint-*")
            if p.is_dir() and p.name.rsplit("-", 1)[-1].isdigit()
        ),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )

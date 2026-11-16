"""Reproducible source-document grouped splitting."""

import random


def grouped_splits(
    source_ids: list[str] | set[str], *, seed: int = 42,
    train_fraction: float = 0.70, val_fraction: float = 0.15,
) -> dict[str, str]:
    """Ensure every source document and its questions stay in exactly one split."""
    ordered = sorted(set(source_ids))
    random.Random(seed).shuffle(ordered)
    if len(ordered) < 3:
        raise ValueError("Grouped train/val/test split requires at least three unique source documents")
    train_end = min(max(1, int(len(ordered) * train_fraction)), len(ordered) - 2)
    val_end = train_end + min(max(1, int(len(ordered) * val_fraction)), len(ordered) - train_end - 1)
    return {source: ("train" if i < train_end else "val" if i < val_end else "test") for i, source in enumerate(ordered)}

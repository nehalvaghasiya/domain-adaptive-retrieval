"""Paired, query-aligned bootstrap intervals for model comparisons."""

from collections.abc import Mapping
from typing import Any

import numpy as np


def paired_bootstrap(
    baseline: Mapping[str, Mapping[str, float]],
    tuned: Mapping[str, Mapping[str, float]],
    metric: str = "ndcg@10",
    *,
    seed: int = 42,
    resamples: int = 1000,
) -> dict[str, Any]:
    """95% percentile interval of paired per-query improvements, not significance proof."""
    qids = sorted(set(baseline) & set(tuned))
    values = np.array(
        [
            tuned[q][metric] - baseline[q][metric]
            for q in qids
            if metric in tuned[q] and metric in baseline[q]
        ],
        dtype=np.float64,
    )
    if not len(values):
        raise ValueError(f"No shared per-query measurements for {metric}")
    generator = np.random.default_rng(seed)
    means = [
        float(values[generator.integers(0, len(values), len(values))].mean())
        for _ in range(resamples)
    ]
    return {
        "metric": metric,
        "queries": len(values),
        "delta": float(values.mean()),
        "paired_bootstrap_95ci": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "resamples": resamples,
    }

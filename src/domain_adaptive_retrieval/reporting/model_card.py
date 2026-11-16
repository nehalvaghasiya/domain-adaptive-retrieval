"""Hugging Face-compatible, evidence-grounded Markdown model cards."""

from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.bundle import RetrievalBundle


def build_model_card(
    *,
    model_type: str,
    parent: str,
    config: ExperimentConfig,
    bundle: RetrievalBundle,
    metrics: dict[str, dict[str, float]] | None = None,
    checkpoint: str | None = None,
    run_id: str = "not-set",
) -> str:
    """Describe real measurements or explicitly say not evaluated; never invent scores."""
    if model_type not in {"embedding", "reranker"}:
        raise ValueError(f"Unknown model type: {model_type}")
    score_lines = [
        f"| {name} | "
        + ", ".join(f"{key}: {value:.4f}" for key, value in sorted(measures.items()))
        + " |"
        for name, measures in sorted((metrics or {}).items())
    ]
    results = (
        "\n".join(["| System | Observed test metrics |", "|---|---|", *score_lines])
        if score_lines
        else "No held-out test scores have been measured for this checkpoint."
    )
    notice = (
        "**Subset experiment:** results are not full-benchmark-comparable."
        if bundle.meta.get("capped_run") or config.max_test_queries
        else "Held-out test evaluation must be independently reproduced."
    )
    upstream_license = "mit" if model_type == "embedding" else "apache-2.0"
    return f"""---
language:
- en
tags:
- sentence-transformers
- retrieval
- domain-adaptation
- {config.domain}
license: {upstream_license}
model-index: []
---
# Domain Adaptive Retrieval — {config.domain} {model_type}

**Parent:** `{parent}`  
**Type:** `{model_type}`  
**Benchmark:** `{bundle.meta.get("dataset", config.domain)}`  
**Protocol:** `{bundle.protocol.value}`  
**Run ID:** `{run_id}`  
**Selected checkpoint:** `{checkpoint or "not trained"}`

## Intended use
Domain-adapted ranking and retrieval experiments. Not for making medical, legal or financial decisions without expert oversight.

## Dataset and splits
Upstream base-model license: `{upstream_license}`. This is **not** a grant of rights to any training data. Original data licensing, rights and annotations are governed by the source dataset, not this repository's MIT license. Train/dev/test boundaries and preprocessing must be preserved. The qrels describe known relevance, not exhaustive truth.

Train queries: {len(bundle.qrels["train"])}; validation queries: {len(bundle.qrels["val"])}; test queries: {len(bundle.qrels["test"])}.

## Training
Base checkpoint: `{parent}`. Objective: `{"MultipleNegativesRankingLoss" if model_type == "embedding" else "BinaryCrossEntropyLoss"}`.
Training epochs: {config.epochs}, batch: {config.batch_size}, accumulation: {config.accumulation}, learning rate: {config.learning_rate}, sequence length: {config.max_sequence_length}, seed: {config.seed}.
Device: `{config.device}`, precision: `{config.precision}`. Explicit mined negatives can contain unjudged false negatives.

## Held-out evaluation
{notice}

{results}

## Limitations and risks
- ESCI metrics cover only judged candidate lists, not whole catalogs.
- Legal character spans are offsets in source documents, and splits are grouped by source.
- CodeSearchNet uses docstring-derived supervision with source code deduplicated across splits.
- Relevance annotation gaps and sampling can affect scores.
- Test data must never influence training, negative mining or checkpoint selection.
- Report inference latency and peak CUDA memory for the exact deployment hardware.

## Reproduction
Use `dar benchmark run` on exactly the dataset manifest and model revisions recorded in the run artifacts. Validate licenses and model card metadata before publishing.
"""


def write_model_card(path: str | Path, **kwargs: Any) -> None:
    """Write a standalone model card inside its checkpoint directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_model_card(**kwargs), encoding="utf-8")

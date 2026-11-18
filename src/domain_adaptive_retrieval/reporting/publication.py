"""Concise report and model-card views of validated experiment records."""

import json
from pathlib import Path
from typing import Any

import yaml

from domain_adaptive_retrieval.reporting.evidence import (
    DATASETS,
    PRIMARY,
    SCOPE,
    TITLES,
    EvaluationRun,
)

PROJECT = "https://github.com/nehalvaghasiya/domain-adaptive-retrieval"
LABELS = {
    "ndcg@10": "nDCG@10",
    "ndcg@5": "nDCG@5",
    "mrr@10": "MRR@10",
    "exact_mrr@10": "Exact MRR@10",
    "exact_hit@10": "Exact Hit@10",
    "recall@100": "Recall@100",
    "recall@50": "Recall@50",
    "recall@10": "Recall@10",
    "legal_character_recall@5": "Character Recall@5",
    "legal_character_precision@5": "Character Precision@5",
    "legal_source_balanced_char_recall@5": "Source-balanced Character Recall@5",
}
QUESTIONS = {
    "finance": "How can a company improve its cash flow?",
    "biomedical": "What are the effects of dietary fiber?",
    "legal": "When can either party terminate the agreement?",
    "ecommerce": "waterproof hiking shoes",
    "code": "Sort a list of records by date.",
}


def inline(value: Any) -> str:
    return "`" + str(value) + "`"


def max_length(run: EvaluationRun) -> int:
    return int(
        run.config.get(
            "max_sequence_length", run.config.get("embedding_max_length", 256)
        )
    )


def outcome(run: EvaluationRun) -> str:
    metric = PRIMARY[run.domain][0]
    return (
        f"Tuned {LABELS[metric]} **{run.tuned[metric]:.4f}** vs "
        f"**{run.base[metric]:.4f}** base "
        f"(Δ {run.tuned[metric] - run.base[metric]:+.4f})."
    )


def metric_table(run: EvaluationRun) -> str:
    """Show every metric, retaining negative changes and uncertainty."""
    principal = PRIMARY[run.domain]
    metrics = sorted(run.base, key=lambda name: (name not in principal, name))
    lines = [
        "| Metric | Base | Tuned | Δ | 95% CI of Δ |",
        "|---|---:|---:|---:|---|",
    ]
    for metric in metrics:
        ci = run.bootstrap.get(metric, {}).get("paired_bootstrap_95ci")
        interval = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—"
        lines.append(
            f"| {LABELS.get(metric, metric)} | {run.base[metric]:.4f} | "
            f"{run.tuned[metric]:.4f} | "
            f"{run.tuned[metric] - run.base[metric]:+.4f} | {interval} |"
        )
    return "\n".join(lines)


def summary_table(runs: list[EvaluationRun], *, reports_prefix: str = "") -> str:
    lines = [
        "| Domain | Model | Test queries | Base nDCG@10 | Tuned nDCG@10 | Δ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        label = f"[{TITLES[run.domain]}]({reports_prefix}{run.domain}.md)"
        if run.status != "evaluated":
            lines.append(
                f"| {label} | {run.model_type} ({run.status}) | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {label} | {run.model_type} | {run.query_count:,} | "
            f"{run.base['ndcg@10']:.4f} | {run.tuned['ndcg@10']:.4f} | "
            f"{run.tuned['ndcg@10'] - run.base['ndcg@10']:+.4f} |"
        )
    return "\n".join(lines)


def highlights(runs: list[EvaluationRun]) -> str:
    lines = []
    for run in runs:
        if run.status != "evaluated" or run.model_type != "embedding":
            continue
        if run.domain == "legal":
            metric = "legal_character_recall@5"
        elif run.domain in {"finance", "biomedical"}:
            metric = "recall@100"
        else:
            continue
        if run.tuned[metric] > run.base[metric]:
            scope = " Document-scoped retrieval." if run.domain == "legal" else ""
            if metric == "recall@100" and run.tuned["ndcg@10"] < run.base["ndcg@10"]:
                scope = " nDCG@10 decreased."
            lines.append(
                f"- **{TITLES[run.domain]} embedding:** {LABELS[metric]} "
                f"{run.base[metric]:.4f} → **{run.tuned[metric]:.4f}** "
                f"(+{100 * (run.tuned[metric] - run.base[metric]):.2f} percentage points).{scope}"
            )
    return "\n".join(sorted(lines, key=lambda line: "Legal" not in line))


def pipeline_table(run: EvaluationRun) -> str:
    metrics = [metric for metric in PRIMARY[run.domain] if metric in run.base]
    lines = [
        "| Saved system | " + " | ".join(LABELS[m] for m in metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for system, scores in run.systems.items():
        lines.append(
            f"| {system} | " + " | ".join(f"{scores[m]:.4f}" for m in metrics) + " |"
        )
    return "\n".join(lines)


def legal_table(run: EvaluationRun) -> str:
    lines = [
        "| Source | Queries | Base Character Recall@5 | Tuned |",
        "|---|---:|---:|---:|",
    ]
    for name, values in sorted(run.legal_sources.items()):
        lines.append(
            f"| {name} | {values['queries']} | {values['base']:.4f} | {values['tuned']:.4f} |"
        )
    return "\n".join(lines)


def domain_report(domain: str, runs: list[EvaluationRun]) -> str:
    sections = [
        f"# {TITLES[domain]} results",
        "[Evaluation overview](README.md) · [Full-precision data](results.json) · [CSV](metrics.csv)",
        SCOPE[domain],
    ]
    for run in runs:
        sections.append(f"## {run.model_type.capitalize()}")
        if run.status != "evaluated":
            sections.append(
                f"**{run.status.capitalize()}.** {run.error or 'No evaluated checkpoint.'}"
            )
            continue
        sections.extend(
            [
                outcome(run),
                f"{run.query_count:,} test queries; {run.corpus_documents:,} documents across "
                f"the loaded dataset. Protocol: {inline(run.protocol)}. "
                f"[Model card](../model-cards/{domain}-{run.model_type}.md).",
                metric_table(run),
            ]
        )
        if run.legal_sources:
            sections.extend(["### Character recall by source", legal_table(run)])
    if any(run.status == "evaluated" for run in runs):
        sections.append(f"![Base and tuned principal metrics](assets/{domain}.svg)")
    if domain == "finance" and len(runs) > 1:
        sections.append(
            "Embedding evaluation used 384 tokens; reranker evaluation used 256. "
            "Each table uses its own run's baseline."
        )
    sections.append(
        "Scores use known judgments; higher is better. Δ = tuned − base. "
        "Intervals are saved query-level paired bootstraps (1,000 resamples); "
        "a dash means no interval was recorded. Full settings, runtime measurements "
        "and source SHA-256 digests are in the JSON export."
    )
    return "\n\n".join(sections) + "\n"


def readme_section(runs: list[EvaluationRun], results_path: str) -> str:
    """Link to the featured model without embedding scores in the project README."""
    run = runs[0]
    return (
        "## Model\n\n"
        f"[Hugging Face model card]({results_path}/../model-cards/"
        f"{run.domain}-{run.model_type}.md) · "
        f"[Evaluation details]({results_path}/{run.domain}.md)\n"
    )


def card_metadata(run: EvaluationRun) -> dict[str, Any]:
    dataset_id, _ = DATASETS[run.domain]
    task = "feature-extraction" if run.model_type == "embedding" else "text-ranking"
    metadata: dict[str, Any] = {
        "language": ["en"],
        "library_name": "sentence-transformers",
        "pipeline_tag": task,
        "base_model": run.parent,
        "base_model_relation": "finetune",
        "license": "mit" if run.model_type == "embedding" else "apache-2.0",
        "tags": ["retrieval", "domain-adaptation", run.model_type, run.domain],
    }
    if dataset_id:
        metadata["datasets"] = [dataset_id]
        dataset: dict[str, Any] = {
            "type": dataset_id,
            "name": f"{run.dataset} ({run.protocol}; {run.query_count} test queries)",
            "split": "test",
            "args": {
                "protocol": run.protocol,
                "test_queries": run.query_count,
                "capped_run": run.capped_run,
                "max_sequence_length": max_length(run),
                "candidate_top_k": run.config.get(
                    "candidate_top_k", run.config.get("ranking_top_k", 100)
                ),
            },
        }
        if run.domain == "code":
            dataset["config"] = "python"
            dataset["args"]["test_corpus_documents"] = run.split_queries["test"]
        if run.model_type == "reranker":
            dataset["args"]["candidate_source"] = run.config["embedding_model"]
        metadata["model-index"] = [
            {
                "name": run.repo_id.split("/")[-1],
                "results": [
                    {
                        "task": {
                            "type": task,
                            "name": "Retrieval"
                            if run.model_type == "embedding"
                            else "Reranking",
                        },
                        "dataset": dataset,
                        "metrics": [
                            {"type": key, "name": LABELS.get(key, key), "value": score}
                            for key, score in run.tuned.items()
                        ],
                    }
                ],
            }
        ]
    return metadata


def usage_code(run: EvaluationRun) -> str:
    query = json.dumps(QUESTIONS[run.domain])
    if run.model_type == "embedding":
        prefix = json.dumps(run.config.get("query_prefix", ""))
        return (
            "from sentence_transformers import SentenceTransformer\n\n"
            'model = SentenceTransformer(".", device="cpu")\n'
            f"model.max_seq_length = {max_length(run)}\n"
            f"query_prefix = {prefix}\n"
            f"query = query_prefix + {query}\n"
            'passages = ["A relevant passage.", "Another candidate passage."]\n'
            "q = model.encode([query], normalize_embeddings=True)\n"
            "docs = model.encode(passages, normalize_embeddings=True)\n"
            "print(q @ docs.T)\n"
        )
    return (
        "from sentence_transformers import CrossEncoder\n\n"
        f'model = CrossEncoder(".", device="cpu", max_length={max_length(run)})\n'
        f"query = {query}\n"
        'passages = ["A relevant passage.", "Another candidate passage."]\n'
        "scores = model.predict([(query, passage) for passage in passages])\n"
        "print(sorted(zip(scores, passages), reverse=True))\n"
    )


def evaluated_model_card(
    run: EvaluationRun, *, packaged: bool = False, public_report: bool = True
) -> str:
    """Describe only validated scores and the exact checkpoint used to obtain them."""
    if run.status != "evaluated":
        raise ValueError("A model card requires a validated evaluated checkpoint")
    cfg, train = run.config, run.training
    _, source_url = DATASETS[run.domain]
    if packaged:
        evaluation_link = "[Evaluation, settings and evidence](evaluation/run.json)"
    elif public_report:
        evaluation_link = (
            f"[Evaluation, settings and evidence](../results/{run.domain}.md)"
        )
    else:
        evaluation_link = "[Evaluation details](#evaluation)"
    objective = train.get("objective", "MultipleNegativesRankingLoss")
    pairs = train.get("training_pairs", train.get("mined_triplets", 0))
    checkpoint = Path(train.get("selected_checkpoint", "unrecorded")).name
    batch = cfg.get("batch_size", cfg.get("train_batch_size", "unrecorded"))
    accumulation = cfg.get("accumulation", cfg.get("grad_accumulation", "unrecorded"))
    val_queries = min(
        cfg.get("max_val_queries", cfg.get("max_dev_queries", 0))
        or run.split_queries["val"],
        run.split_queries["val"],
    )
    weight_dtype = ", ".join(run.model["saved_dtypes"])
    specs = f"{run.model['parameters']:,} parameters; saved weights {weight_dtype}"
    if run.model_type == "embedding":
        specs += f"; {run.model['embedding_dimension']}-dimensional vectors"
    splits = " / ".join(
        f"{run.split_queries[split]:,}" for split in ("train", "val", "test")
    )
    precision = cfg.get("precision", run.hardware.get("mixed_precision", "unrecorded"))
    provenance = (
        "Recorded checkpoint hash verified."
        if run.checkpoint_fingerprint
        else "Historical checkpoint hash unrecorded; current source digests provided."
    )
    strategy = cfg.get("negative_strategy", "unrecorded")
    peak = train.get("gpu_peak_allocated_gib")
    if peak is None and "peak_cuda_memory_bytes" in train:
        peak = train["peak_cuda_memory_bytes"] / 1024**3
    runtime = f"Training {train['training_seconds']:.1f} s"
    if peak is not None:
        runtime += f"; peak allocated CUDA memory {peak:.2f} GiB"
    runtime += f"; GPU {run.hardware.get('gpu_name', 'model unrecorded')}."
    if run.domain == "legal":
        limitation = "Custom source-document split; searches within one document, not a legal library. Character spans are unioned to avoid double counting."
    elif run.domain == "code":
        limitation = "Python only; 15,000/500/500 split caps. Docstrings are removed from code and code is deduplicated across splits. The 500-document test pool is not the full CodeSearchNet benchmark."
    elif run.domain == "ecommerce":
        limitation = "Grades E/S/C/I = 3/2/0/0; Exact MRR counts E only. Results cover judged product lists, not whole-catalog retrieval."
    else:
        limitation = "Full-corpus evaluation after recorded empty-text filtering. Sparse judgments can miss relevant documents."
    candidate_note = (
        "Both rerankers receive the same top-100 candidates from the base BGE model. "
        "Scores are relevance logits, not calibrated probabilities."
        if run.model_type == "reranker"
        else "Base and tuned embeddings retrieve independently from the same permitted scope."
    )
    environment = "; ".join(f"{key}={value}" for key, value in run.versions.items())
    return (
        "---\n"
        + yaml.safe_dump(card_metadata(run), sort_keys=False, allow_unicode=True)
        + "---\n\n"
        + f"# {TITLES[run.domain]} {run.model_type}\n\n"
        + f"Prepared model ID: {inline(run.repo_id)}.\n\n"
        + f"{outcome(run)} Research checkpoint for English domain retrieval.\n\n"
        + f"Parent: [{run.parent}](https://huggingface.co/{run.parent}). "
        + f"{specs}. Evaluated sequence limit: **{max_length(run)} tokens**.\n\n"
        + "## Use\n\n"
        + "Install `sentence-transformers>=5.1,<6`; run from the downloaded model directory.\n\n"
        + "```python\n"
        + usage_code(run)
        + "```\n\n"
        + "## Evaluation\n\n"
        + f"[{run.dataset}]({source_url}); **{run.query_count:,} test queries**. "
        + SCOPE[run.domain]
        + " "
        + candidate_note
        + "\n\n"
        + metric_table(run)
        + "\n\n"
        + "Higher is better; Δ = tuned − base. CIs are saved query-level paired "
        + "bootstrap intervals (1,000 resamples); — means unrecorded. "
        + "These results are specific to this protocol.\n\n"
        + "## Training\n\n"
        + f"Train / validation / test queries: **{splits}**. "
        + f"{pairs:,} training examples; {inline(objective)}; {strategy} negatives.\n\n"
        + "| Setting | Value |\n|---|---|\n"
        + f"| Epochs / learning rate | {cfg.get('epochs')} / {cfg.get('learning_rate')} |\n"
        + f"| Batch / accumulation | {batch} / {accumulation} |\n"
        + f"| Seed / training precision | {cfg.get('seed')} / {precision} |\n"
        + f"| Selected checkpoint | {checkpoint} |\n"
        + f"| Selection | Validation nDCG@10 on {val_queries:,} queries |\n\n"
        + runtime
        + " Saved dtype and training precision are distinct.\n\n"
        + "## Limits and attribution\n\n"
        + limitation
        + " Mined unjudged negatives can be false negatives. "
        + "Intended for retrieval experiments; downstream answer quality and deployment "
        + "latency are unmeasured. "
        + (
            "Recorded latency is component-only and excludes embedding encoding. "
            if run.latency
            else ""
        )
        + f"Base-model license: {'MIT' if run.model_type == 'embedding' else 'Apache-2.0'}; "
        + "dataset rights and attribution remain governed by the linked dataset sources.\n\n"
        + f"{evaluation_link} · [Source and reproduction]({PROJECT}). "
        + provenance
        + " Upstream revisions were not pinned; no exact historical rerun is claimed.\n\n"
        + "<details>\n<summary>Recorded environment</summary>\n\n"
        + inline(environment)
        + "\n\n</details>\n"
    )

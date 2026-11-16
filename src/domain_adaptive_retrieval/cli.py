"""Small discoverable command line interface; heavy ML imports are lazy."""

import json
from pathlib import Path
from typing import Any

import typer

from domain_adaptive_retrieval.config.loader import load_config

app = typer.Typer(
    no_args_is_help=True,
    help="Domain-adaptive embedding and reranker retrieval experiments",
)
config_app = typer.Typer(help="Inspect and validate experiment configurations")
hardware_app = typer.Typer(help="Check CPU/CUDA requirements")
data_app = typer.Typer(help="Load, prepare and audit a domain dataset")
train_app = typer.Typer(help="Train one embedding model or reranker")
evaluate_app = typer.Typer(help="Evaluate one checkpoint on held-out labels")
benchmark_app = typer.Typer(help="Compare BM25, base and adapted models (S0–S6)")
report_app = typer.Typer(help="Produce readable benchmark summaries")
publish_app = typer.Typer(help="Explicit, permission-checked Hugging Face publishing")
for name, subapp in (
    ("config", config_app),
    ("hardware", hardware_app),
    ("data", data_app),
    ("train", train_app),
    ("evaluate", evaluate_app),
    ("benchmark", benchmark_app),
    ("report", report_app),
    ("publish", publish_app),
):
    app.add_typer(subapp, name=name)


def _experiment(domain: str, profile: str, config: Path | None) -> Any:
    from domain_adaptive_retrieval.config.presets import domain_config

    domain_path = config if config is not None else domain_config(domain)
    profile_path = Path(f"configs/experiments/{profile}.yaml") if profile else None
    return load_config(domain_path, profile=profile_path, domain=domain)


@config_app.command("validate")
def validate_config(
    config: Path = typer.Option(..., exists=True), profile: Path | None = None
) -> None:
    """Validate YAML and display resolved device and experiment scope."""
    cfg = load_config(config, profile=profile)
    typer.echo(cfg.model_dump_json(indent=2))


@hardware_app.command("check")
def hardware_check(device: str = "cpu", require_gpu: bool = False) -> None:
    """Fail if a CUDA device was selected but is not available."""
    from domain_adaptive_retrieval.models.device import check_device

    typer.echo(json.dumps(check_device(device, require_gpu=require_gpu), indent=2))


@data_app.command("prepare")
def data_prepare(
    domain: str = "finance",
    output: Path = Path("data/processed"),
    config: Path | None = None,
) -> None:
    """Download/load and validate data, then write an audit and split manifest."""
    from domain_adaptive_retrieval.workflows.prepare import prepare_data

    cfg = _experiment(domain, "", config)
    bundle = prepare_data(cfg, output / cfg.domain)
    typer.echo(
        f"Validated {cfg.domain}: {len(bundle.corpus)} documents, {len(bundle.qrels['test'])} test queries"
    )


@data_app.command("audit")
def data_audit(domain: str = "finance", config: Path | None = None) -> None:
    """Describe corpus/qrel integrity without model loading or training."""
    from domain_adaptive_retrieval.data.registry import load_domain
    from domain_adaptive_retrieval.data.validation import validate_bundle

    typer.echo(
        json.dumps(
            validate_bundle(load_domain(_experiment(domain, "", config))), indent=2
        )
    )


@train_app.command("embedder")
def train_embedder(
    domain: str = "finance",
    profile: str = "t4",
    output: Path = Path("outputs"),
    config: Path | None = None,
) -> None:
    """Fine-tune BGE with MNRL and choose the best validation checkpoint."""
    from domain_adaptive_retrieval.workflows.train_embedder import train_embedding

    cfg = _experiment(domain, profile, config)
    typer.echo(
        f"Saved embedding checkpoint: {train_embedding(cfg, output / domain / 'embedding')}"
    )


@train_app.command("reranker")
def train_reranker(
    domain: str = "finance",
    profile: str = "t4",
    output: Path = Path("outputs"),
    config: Path | None = None,
) -> None:
    """Fine-tune MiniLM with BCE over mined and labeled train examples."""
    from domain_adaptive_retrieval.workflows.train_reranker import (
        train_reranker as execute,
    )

    cfg = _experiment(domain, profile, config)
    typer.echo(
        f"Saved reranker checkpoint: {execute(cfg, output / domain / 'reranker')}"
    )


@evaluate_app.command("embedder")
def evaluate_embedder(
    checkpoint: Path = typer.Option(...),
    domain: str = "finance",
    profile: str = "t4",
    output: Path = Path("outputs/evaluate_embedding"),
    config: Path | None = None,
) -> None:
    """Evaluate dense base vs tuned retrieval without training or reranker."""
    from domain_adaptive_retrieval.data.registry import load_domain
    from domain_adaptive_retrieval.evaluation.evaluator import evaluate_predictions
    from domain_adaptive_retrieval.models.embedder import EmbeddingModel
    from domain_adaptive_retrieval.reporting.artifacts import (
        write_json,
        write_predictions,
    )
    from domain_adaptive_retrieval.retrieval.dense import DenseRetriever

    cfg = _experiment(domain, profile, config)
    bundle = load_domain(cfg)
    if not checkpoint.is_dir():
        raise typer.BadParameter(f"Fine-tuned checkpoint missing: {checkpoint}")
    models = {"base": cfg.embedding_model, "tuned": str(checkpoint)}
    output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}
    for label, name in models.items():
        ranker = DenseRetriever(
            EmbeddingModel(name, device=cfg.device, max_length=cfg.max_sequence_length),
            device=cfg.device,
            query_prefix=cfg.query_prefix,
        )
        qids = bundle.split_queries("test")[: cfg.max_test_queries or None]
        pred = ranker.search(bundle, qids, split="test", top_k=cfg.candidate_top_k)
        metrics[label], _ = evaluate_predictions(
            bundle, pred, split="test", query_ids=qids
        )
        write_predictions(output / f"predictions_{label}.jsonl", pred, bundle)
    write_json(output / "metrics.json", metrics)
    typer.echo(json.dumps(metrics, indent=2))


@evaluate_app.command("reranker")
def evaluate_reranker(
    checkpoint: Path = typer.Option(...),
    domain: str = "finance",
    profile: str = "t4",
    output: Path = Path("outputs/evaluate_reranker"),
    config: Path | None = None,
) -> None:
    """Evaluate two rerankers against the same frozen base embedding candidates."""
    from domain_adaptive_retrieval.data.registry import load_domain
    from domain_adaptive_retrieval.evaluation.evaluator import evaluate_predictions
    from domain_adaptive_retrieval.models.embedder import EmbeddingModel
    from domain_adaptive_retrieval.models.reranker import RerankerModel
    from domain_adaptive_retrieval.reporting.artifacts import (
        write_json,
        write_predictions,
    )
    from domain_adaptive_retrieval.retrieval.candidates import save_candidates
    from domain_adaptive_retrieval.retrieval.dense import DenseRetriever
    from domain_adaptive_retrieval.retrieval.rerank import rerank_candidates

    cfg = _experiment(domain, profile, config)
    bundle = load_domain(cfg)
    if not checkpoint.is_dir():
        raise typer.BadParameter(f"Fine-tuned checkpoint missing: {checkpoint}")
    qids = bundle.split_queries("test")[: cfg.max_test_queries or None]
    dense = DenseRetriever(
        EmbeddingModel(cfg.embedding_model, device=cfg.device),
        device=cfg.device,
        query_prefix=cfg.query_prefix,
    )
    candidates = dense.search(bundle, qids, split="test", top_k=cfg.candidate_top_k)
    output.mkdir(parents=True, exist_ok=True)
    save_candidates(output / "frozen_candidates.json", candidates)
    scores: dict[str, dict[str, float]] = {}
    for label, name in (("base", cfg.reranker_model), ("tuned", str(checkpoint))):
        pred = rerank_candidates(
            bundle, candidates, RerankerModel(name, device=cfg.device)
        )
        scores[label], _ = evaluate_predictions(
            bundle, pred, split="test", query_ids=qids
        )
        write_predictions(output / f"predictions_{label}.jsonl", pred, bundle)
    write_json(output / "metrics.json", scores)
    typer.echo(json.dumps(scores, indent=2))


@benchmark_app.command("run")
def benchmark_run(
    embedder: Path = typer.Option(...),
    reranker: Path = typer.Option(...),
    domain: str = "finance",
    profile: str = "t4",
    output: Path = Path("outputs/benchmark"),
    config: Path | None = None,
) -> None:
    """Run all S0–S6 with frozen candidate lists and held-out qrels."""
    from domain_adaptive_retrieval.workflows.benchmark import run_benchmark

    cfg = _experiment(domain, profile, config)
    outcome = run_benchmark(cfg, embedder, reranker, output / domain)
    typer.echo(json.dumps(outcome["metrics"], indent=2))


@report_app.command("build")
def report_build(run: Path = typer.Option(..., exists=True)) -> None:
    """Create a readable summary from a saved benchmark artifact."""
    from domain_adaptive_retrieval.reporting.comparison import write_comparison

    record = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    target = run / "comparison.md"
    write_comparison(
        target, metrics, protocol=record["protocol"], capped=record["capped_run"]
    )
    typer.echo(f"Wrote {target}")


@publish_app.command("model")
def publish_model(
    run: Path = typer.Option(...),
    repo_id: str = typer.Option(...),
    private: bool = True,
    confirm: bool = False,
) -> None:
    """Publish trained weights only with deliberate --confirm and rights checks."""
    from domain_adaptive_retrieval.reporting.publish import publish_checkpoint

    checkpoint = run / "model" if (run / "model").is_dir() else run
    typer.echo(
        str(
            publish_checkpoint(
                checkpoint, repo_id=repo_id, private=private, confirm=confirm
            )
        )
    )


@app.command("smoke")
def smoke(
    offline: bool = True,
    config: Path | None = None,
    output: Path = Path("outputs/smoke"),
) -> None:
    """Run six deterministic dataset workflows and S0–S6 without downloads."""
    from domain_adaptive_retrieval.workflows.smoke import run_smoke

    if not offline:
        raise typer.BadParameter(
            "This command is intentionally offline; use benchmark run for real models"
        )
    if config is not None:
        load_config(config)
    outcome = run_smoke(output)
    typer.echo(json.dumps(outcome, indent=2))


def main() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    main()

"""Read and verify saved experiments without loading datasets or changing artifacts."""

import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import BaseModel, Field

from domain_adaptive_retrieval.models.checkpoint import fingerprint_path
from domain_adaptive_retrieval.retrieval.candidates import load_candidates

DOMAINS = ("finance", "biomedical", "legal", "ecommerce", "code")
TITLES = {
    "finance": "Finance",
    "biomedical": "Biomedical",
    "legal": "Legal",
    "ecommerce": "E-commerce",
    "code": "Code",
}
DATASETS = {
    "finance": ("BeIR/fiqa", "https://huggingface.co/datasets/BeIR/fiqa"),
    "biomedical": ("BeIR/nfcorpus", "https://huggingface.co/datasets/BeIR/nfcorpus"),
    "legal": (
        "awinml/legalbench-rag",
        "https://huggingface.co/datasets/awinml/legalbench-rag",
    ),
    "ecommerce": ("", "https://github.com/amazon-science/esci-data"),
    "code": (
        "claudios/code_search_net",
        "https://huggingface.co/datasets/claudios/code_search_net",
    ),
}
SCOPE = {
    "finance": "Full corpus after empty-text filtering.",
    "biomedical": "Full corpus.",
    "legal": "Document-scoped retrieval; custom split grouped by source document.",
    "ecommerce": "Amazon ESCI US Task 1; only products judged for each query.",
    "code": "CodeSearchNet Python subset; 500 test queries search 500 test documents.",
}
PRIMARY = {
    "finance": ("ndcg@10", "recall@100"),
    "biomedical": ("ndcg@10", "recall@100"),
    "legal": ("legal_character_recall@5", "ndcg@10"),
    "ecommerce": ("ndcg@10", "exact_mrr@10"),
    "code": ("mrr@10", "ndcg@10"),
}


class _EvidenceFile(BaseModel):
    """A source-relative path and independently computed file digest."""

    path: str
    sha256: str
    bytes: int


class EvaluationRun(BaseModel):
    """One experiment; original variant names retain their original meaning."""

    id: str
    domain: str
    model_type: str
    repo_id: str
    source: str
    status: str = "evaluated"
    error: str | None = None
    artifact_format: str = ""
    dataset: str = ""
    protocol: str = ""
    parent: str = ""
    query_count: int = 0
    corpus_documents: int = 0
    split_queries: dict[str, int] = Field(default_factory=dict)
    capped_run: bool = False
    base_label: str = "base"
    tuned_label: str = "tuned"
    systems: dict[str, dict[str, float]] = Field(default_factory=dict)
    bootstrap: dict[str, dict[str, Any]] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    versions: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)
    latency: dict[str, Any] = Field(default_factory=dict)
    legal_sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    checkpoint_fingerprint: str | None = None
    notes: list[str] = Field(default_factory=list)
    evidence: list[_EvidenceFile] = Field(default_factory=list)

    @property
    def base(self) -> dict[str, float]:
        return self.systems.get(self.base_label, {})

    @property
    def tuned(self) -> dict[str, float]:
        return self.systems.get(self.tuned_label, {})


def read_json(path: Path) -> dict[str, Any]:
    """Require a JSON object, including for otherwise optional metadata."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_spec(directory: Path) -> dict[str, Any]:
    """Inspect tensor headers only; neither Torch nor model downloads are needed."""
    weights = directory / "model.safetensors"
    with weights.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size > min(weights.stat().st_size - 8, 100 * 1024 * 1024):
            raise ValueError(f"Invalid safetensors header: {weights}")
        header = json.loads(stream.read(header_size))
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    config = read_json(directory / "config.json")
    return {
        "parameters": sum(math.prod(tensor["shape"]) for tensor in tensors),
        "saved_dtypes": sorted({tensor["dtype"] for tensor in tensors}),
        "architecture": config.get("architectures", []),
        "embedding_dimension": config.get("hidden_size"),
        "weights_bytes": weights.stat().st_size,
    }


def _close(actual: float, expected: float, description: str) -> None:
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, abs_tol=1e-10, rel_tol=1e-10
    ):
        raise ValueError(f"Inconsistent {description}: {actual} != {expected}")


def _validate_metrics(run: EvaluationRun, per_query: dict[str, Any]) -> None:
    reference = set(per_query[run.base_label])
    if not reference or len(reference) != run.query_count:
        raise ValueError(f"Query count mismatch: {run.id}")
    for variant, metrics in run.systems.items():
        rows = per_query[variant]
        if set(rows) != reference:
            raise ValueError(f"Query alignment mismatch: {run.id}/{variant}")
        for metric, score in metrics.items():
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid metric: {run.id}/{variant}/{metric}")
            if metric == "legal_source_balanced_char_recall@5":
                groups: dict[str, list[float]] = defaultdict(list)
                for qid, values in rows.items():
                    groups[qid.split(":")[0]].append(values["legal_character_recall@5"])
                mean = fmean(fmean(values) for values in groups.values())
            else:
                mean = fmean(values[metric] for values in rows.values())
            _close(score, mean, f"{run.id}/{variant}/{metric}")
    if run.base.keys() != run.tuned.keys():
        raise ValueError(f"Metric alignment mismatch: {run.id}")
    for metric, estimate in run.bootstrap.items():
        _close(estimate["delta"], run.tuned[metric] - run.base[metric], metric)
        lo, hi = estimate["paired_bootstrap_95ci"]
        if (
            estimate["queries"] != len(reference)
            or estimate["resamples"] <= 0
            or not math.isfinite(lo)
            or not math.isfinite(hi)
            or lo > hi
        ):
            raise ValueError(f"Invalid bootstrap estimate: {run.id}/{metric}")
    if run.domain == "legal":
        for qid in sorted(reference):
            source = qid.split(":")[0]
            run.legal_sources.setdefault(
                source, {"queries": 0, "base": [], "tuned": []}
            )
            group = run.legal_sources[source]
            group["queries"] += 1
            for name, label in (("base", run.base_label), ("tuned", run.tuned_label)):
                group[name].append(per_query[label][qid]["legal_character_recall@5"])
        for group in run.legal_sources.values():
            group.update(base=fmean(group["base"]), tuned=fmean(group["tuned"]))


def _predictions(path: Path, qids: set[str]) -> dict[str, set[str]]:
    documents: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = row["query_id"]
            did = row.get("document_id", row.get("doc_id"))
            seen = documents[qid]
            if (
                not isinstance(did, str)
                or did in seen
                or row["rank"] != len(seen) + 1
                or not math.isfinite(row["score"])
            ):
                raise ValueError(f"Invalid prediction: {path}/{qid}")
            seen.add(did)
    if set(documents) != qids:
        raise ValueError(f"Prediction query coverage mismatch: {path}")
    return dict(documents)


def _validate_predictions(directory: Path, run: EvaluationRun, qids: set[str]) -> None:
    candidates: dict[str, dict[str, set[str]]] = {}
    for variant in run.systems:
        candidates[variant] = _predictions(
            directory / f"predictions_{variant}.jsonl", qids
        )
    if run.model_type == "reranker":
        frozen = load_candidates(directory / "frozen_candidates.json")
        frozen_ids = {qid: {row[0] for row in rows} for qid, rows in frozen.items()}
        if any(candidates[label] != frozen_ids for label in ("base", "tuned")):
            raise ValueError(f"Frozen candidate mismatch: {run.id}")
    elif run.artifact_format == "notebook":
        for dense, reranked in (
            ("S1_base_embedding", "S3_base_embedding_base_reranker"),
            ("S2_tuned_embedding", "S5_tuned_embedding_base_reranker"),
        ):
            if reranked in candidates and candidates[dense] != candidates[reranked]:
                raise ValueError(f"Pipeline candidate mismatch: {run.id}/{reranked}")


def _metadata(directory: Path, run: EvaluationRun) -> None:
    audit = read_json(directory / "dataset_audit.json")
    evaluation_audit = optional_json(directory / "evaluation/dataset_audit.json")
    run.dataset = audit["dataset"]
    run.protocol = audit["protocol"]
    run.split_queries = audit.get("split_queries", audit.get("query_counts", {}))
    run.query_count = evaluation_audit.get(
        "evaluated_queries", run.split_queries.get("test", 0)
    )
    run.corpus_documents = audit["corpus_documents"]
    run.capped_run = bool(
        evaluation_audit.get("capped_run", audit.get("capped_run", False))
    )
    newer = run.artifact_format == "suite"
    run.config = read_json(
        directory / ("effective_config.json" if newer else "config.json")
    )
    run.parent = run.config[
        ("embedding_model" if newer else "base_embedding")
        if run.model_type == "embedding"
        else ("reranker_model" if newer else "base_reranker")
    ]
    run.training = read_json(directory / "training_stats.json")
    run.versions = read_json(
        directory / ("environment_versions.json" if newer else "versions.json")
    )
    run.hardware = optional_json(directory / "gpu_runtime.json")
    run.latency = optional_json(directory / "evaluation/latency.json")
    run.model = model_spec(directory / "model")
    run.notes.append(
        "Base-model and dataset revisions were not pinned in the saved configuration."
    )
    if not newer:
        run.notes.append("Historical checkpoint fingerprint was not recorded.")
    else:
        progress = read_json(directory / "evaluation/evaluation_progress.json")
        signature = progress["signature"]
        config_hash = hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in run.config.items()
                    if key != "resume_checkpoint"
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if (
            not progress.get("complete")
            or set(progress["variants"]) != {"base", "tuned"}
            or signature["kind"] != run.model_type
            or signature["dataset"] != audit["fingerprint"]
            or signature["dataset"] != evaluation_audit["fingerprint"]
            or signature["config"] != config_hash
            or read_json(directory / "evaluation/effective_config.json") != run.config
        ):
            raise ValueError(f"Incomplete or mismatched evaluation: {run.id}")
        for files in progress["variants"].values():
            for name, size in files.items():
                if (directory / "evaluation" / name).stat().st_size != size:
                    raise ValueError(f"Evaluation file size mismatch: {run.id}/{name}")
        run.checkpoint_fingerprint = fingerprint_path(directory / "model")
        if run.checkpoint_fingerprint != signature["checkpoint"]:
            raise ValueError(f"Checkpoint fingerprint mismatch: {run.id}")
        for label in ("base", "tuned"):
            result = read_json(directory / "evaluation" / f"result_{label}.json")
            if result["metrics"] != run.systems[label]:
                raise ValueError(f"Variant result mismatch: {run.id}/{label}")
        if (
            optional_json(directory / "run_manifest.json").get("status")
            == "trained_not_yet_tested"
        ):
            run.notes.append(
                "Training manifest predates evaluation; completed evaluation was verified."
            )


def source_files(directory: Path) -> list[Path]:
    """Keep source metadata, evaluation evidence and inference files, not trainer state."""
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix == ".json"]
    for folder in ("evaluation", "model"):
        files.extend(p for p in (directory / folder).rglob("*") if p.is_file())
    return sorted(p for p in files if ":Zone.Identifier" not in p.name)


def _source_evidence(directory: Path) -> list[_EvidenceFile]:
    return [
        _EvidenceFile(
            path=path.relative_to(directory).as_posix(),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
        )
        for path in source_files(directory)
    ]


def load_run(directory: Path, namespace: str) -> EvaluationRun:
    """Fail on conflicting evidence rather than swapping variants or guessing scores."""
    domain, kind = directory.parent.name, directory.name
    run = EvaluationRun(
        id=f"{domain}/{kind}",
        domain=domain,
        model_type=kind,
        repo_id=f"{namespace}/dar-{domain}-{kind}-v1",
        source=f"{directory.parent.parent.name}/{domain}/{kind}",
    )
    if not (directory / "evaluation/metrics.json").is_file():
        state = optional_json(directory / "job_state.json")
        run.status = "failed" if state.get("status") == "failed" else "unavailable"
        error = state.get("error") or ""
        run.error = (
            "Missing legal corpus file during data loading."
            if domain == "legal"
            and "FileNotFoundError" in error
            and "/corpus/" in error
            else error or None
        )
        if directory.is_dir():
            run.evidence = _source_evidence(directory)
        return run
    run.systems = read_json(directory / "evaluation/metrics.json")
    if "base" in run.systems:
        run.artifact_format = "suite"
        per_query_name, bootstrap_name = "per_query.json", "paired_bootstrap.json"
    else:
        run.artifact_format = "notebook"
        run.base_label, run.tuned_label = "S1_base_embedding", "S2_tuned_embedding"
        per_query_name, bootstrap_name = (
            "per_query_metrics.json",
            "bootstrap_delta.json",
        )
    if run.base_label not in run.systems or run.tuned_label not in run.systems:
        raise ValueError(f"Missing base or tuned variant: {run.id}")
    run.bootstrap = optional_json(directory / "evaluation" / bootstrap_name)
    _metadata(directory, run)
    per_query = read_json(directory / "evaluation" / per_query_name)
    _validate_metrics(run, per_query)
    _validate_predictions(directory / "evaluation", run, set(per_query[run.base_label]))
    run.evidence = _source_evidence(directory)
    return run


def scan_runs(root: Path, namespace: str) -> list[EvaluationRun]:
    """Include all ten slots; synthetic smoke directories are never imported."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", namespace):
        raise ValueError("Invalid Hugging Face namespace")
    if not root.is_dir():
        raise FileNotFoundError(root)
    runs = []
    for domain in DOMAINS:
        for kind in ("embedding", "reranker"):
            try:
                run = load_run(root / domain / kind, namespace)
            except (ValueError, KeyError, OSError, TypeError, struct.error) as exc:
                run = EvaluationRun(
                    id=f"{domain}/{kind}",
                    domain=domain,
                    model_type=kind,
                    repo_id=f"{namespace}/dar-{domain}-{kind}-v1",
                    source=f"{root.name}/{domain}/{kind}",
                    status="invalid",
                    error=str(exc),
                )
            runs.append(run)
    return runs

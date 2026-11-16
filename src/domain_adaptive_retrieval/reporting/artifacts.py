"""Atomic, reproducible experiment artifact writing."""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.retrieval.candidates import Predictions


def write_json(path: str | Path, data: Any) -> None:
    """Write indented JSON atomically to avoid corrupt partial manifests."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(
        dir=destination.parent, prefix=".dar_", suffix=".json"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                data, stream, indent=2, ensure_ascii=False, default=str, allow_nan=False
            )
            stream.write("\n")
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def dataset_fingerprint(bundle: RetrievalBundle) -> str:
    """SHA of sorted corpus, queries, qrels, protocol and scope metadata."""
    source = {
        "corpus": sorted(bundle.corpus.items()),
        "queries": sorted(bundle.queries.items()),
        "qrels": {
            split: sorted(
                (qid, sorted(judges.items()))
                for qid, judges in bundle.qrels[split].items()
            )
            for split in sorted(bundle.qrels)
        },
        "pools": sorted(bundle.pools.items()),
        "split_docs": sorted(
            (split, sorted(ids)) for split, ids in bundle.split_docs.items()
        ),
        "scopes": sorted(bundle.scopes.items()),
        "protocol": bundle.protocol.value,
    }
    return hashlib.sha256(
        json.dumps(source, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def write_predictions(
    path: str | Path,
    predictions: Predictions,
    bundle: RetrievalBundle,
    *,
    split: str = "test",
) -> None:
    """Export one JSONL record per predicted query, document and score."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for qid in sorted(predictions):
            for rank, (did, score) in enumerate(predictions[qid], 1):
                entry = {
                    "query_id": qid,
                    "document_id": did,
                    "rank": rank,
                    "score": score,
                    "judged_relevance": bundle.qrels[split].get(qid, {}).get(did),
                }
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_benchmark(
    output: str | Path, bundle: RetrievalBundle, result: dict[str, Any]
) -> Path:
    """Write all system metrics, candidate fingerprints and scored rankings."""
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    for key in ("metrics", "per_query", "candidate_hashes", "candidate_count"):
        write_json(path / f"{key}.json", result[key])
    write_json(
        path / "dataset_audit.json",
        {
            "fingerprint": dataset_fingerprint(bundle),
            "dataset": bundle.meta.get("dataset"),
            "protocol": bundle.protocol.value,
            "data_quality": bundle.meta.get("data_quality", {}),
            "queries_by_split": {
                split: len(bundle.qrels[split]) for split in bundle.qrels
            },
            "capped_run": result["capped_run"],
            "evaluated_queries": result["query_count"],
        },
    )
    write_json(path / "split_manifest.json", bundle.meta.get("split_manifest", {}))
    with (path / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        keys = sorted(
            {metric for group in result["metrics"].values() for metric in group}
        )
        writer = csv.DictWriter(file, fieldnames=["system", *keys])
        writer.writeheader()
        for system, measures in sorted(result["metrics"].items()):
            writer.writerow({"system": system, **measures})
    for system, predictions in result["predictions"].items():
        write_predictions(
            path / f"predictions_{system}.jsonl",
            predictions,
            bundle,
            split=result["split"],
        )
    write_json(
        path / "run_manifest.json",
        {
            "status": "synthetic"
            if bundle.meta.get("synthetic")
            else "actual_inference",
            "data_fingerprint": dataset_fingerprint(bundle),
            "evaluation_split": result["split"],
            "protocol": result["protocol"],
            "capped_run": result["capped_run"],
            "system_codes": sorted(result["metrics"]),
            "candidate_hashes": result["candidate_hashes"],
        },
    )
    return path

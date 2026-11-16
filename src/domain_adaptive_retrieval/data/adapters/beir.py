"""Official BEIR FiQA and NFCorpus loaders, with the empty-qrel regression fixed."""

import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.cleaning import clean_text
from domain_adaptive_retrieval.data.io import read_jsonl

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def _download_beir(name: str, root: Path) -> None:
    """Fetch an official dataset archive, rejecting unsafe members."""
    archive = root.parent / f"{name}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen(f"{BEIR_BASE}/{name}.zip", timeout=180) as response:
            with archive.with_suffix(".zip.part").open("wb") as sink:
                shutil.copyfileobj(response, sink)
        archive.with_suffix(".zip.part").replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            relative = Path(item.filename)
            if relative.is_absolute() or ".." in relative.parts or not str((root.parent / relative).resolve()).startswith(str(root.parent.resolve()) + "/"):
                raise ValueError(f"Unsafe BEIR archive member: {item.filename}")
        zipped.extractall(root.parent)


def load_beir(domain: str, cache_dir: str | Path) -> RetrievalBundle:
    """Load BEIR with stable, string-typed IDs and audited empty-document exclusions.

    Qrels mentioning filtered *known empty* records are dropped, while a genuinely
    absent ID is a hard error. Query `461` is the classic FiQA regression case.
    """
    if domain not in {"finance", "biomedical"}:
        raise ValueError(f"BEIR supports finance and biomedical, not {domain!r}")
    name = "fiqa" if domain == "finance" else "nfcorpus"
    root = Path(cache_dir) / "beir" / name
    expected = [root / "corpus.jsonl", root / "queries.jsonl", *(root / "qrels" / f"{name}.tsv" for name in ("train", "dev", "test"))]
    if not all(item.is_file() for item in expected):
        _download_beir(name, root)
    missing = [str(item) for item in expected if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete BEIR/{name} archive, missing: {missing}")
    bundle = RetrievalBundle(meta={"dataset": f"BEIR/{name}", "protocol": "full_corpus"})
    empty_documents: set[str] = set()
    for row in read_jsonl(root / "corpus.jsonl"):
        did = str(row["_id"])
        content = "\n".join(filter(None, (clean_text(row.get("title")), clean_text(row.get("text")))))
        if content:
            bundle.corpus[did] = content
        else:
            empty_documents.add(did)
    dropped_queries: set[str] = set()
    for row in read_jsonl(root / "queries.jsonl"):
        qid, text = str(row["_id"]), clean_text(row.get("text"))
        if text:
            bundle.queries[qid] = text
        else:
            dropped_queries.add(qid)
    quality: dict[str, object] = {
        "filtered_empty_documents": len(empty_documents),
        "filtered_empty_queries": len(dropped_queries),
        "filtered_qrels": {},
        "queries_without_remaining_judgments": {},
    }
    dropped_per_split: dict[str, int] = {}
    no_judgment: dict[str, int] = {}
    for split, source_name in (("train", "train"), ("val", "dev"), ("test", "test")):
        skipped = 0
        present_queries: set[str] = set()
        with (root / "qrels" / f"{source_name}.tsv").open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                qid, did = str(row["query-id"]), str(row["corpus-id"])
                present_queries.add(qid)
                if did not in bundle.corpus and did not in empty_documents:
                    raise ValueError(f"BEIR/{name} split={split} query={qid}: qrels references truly missing doc {did!r}")
                if qid not in bundle.queries and qid not in dropped_queries:
                    raise ValueError(f"BEIR/{name} split={split}: missing query {qid!r}")
                if did in empty_documents or qid in dropped_queries:
                    skipped += 1
                    continue
                bundle.qrels[split].setdefault(qid, {})[did] = float(row["score"])
        dropped_per_split[split] = skipped
        no_judgment[split] = len(present_queries - bundle.qrels[split].keys())
    quality["filtered_qrels"] = dropped_per_split
    quality["queries_without_remaining_judgments"] = no_judgment
    bundle.meta["data_quality"] = quality
    return bundle

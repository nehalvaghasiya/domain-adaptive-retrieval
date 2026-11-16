"""Generic user-supplied retrieval data in JSONL/CSV/TSV/Parquet tables."""

from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.cleaning import clean_text
from domain_adaptive_retrieval.data.io import read_table


def _table(root: Path, name: str, *, optional: bool = False) -> list[dict[str, Any]]:
    for suffix in (".jsonl", ".tsv", ".csv", ".parquet"):
        path = root / f"{name}{suffix}"
        if path.is_file():
            return read_table(path)
    if optional:
        return []
    raise FileNotFoundError(f"Missing {name}.jsonl/.tsv/.csv/.parquet in {root}")


def load_custom(root: str | Path, *, protocol: str = "full_corpus") -> RetrievalBundle:
    """Load corpus, queries and qrels tables, plus optional legal/ESCI fields.

    Columns: corpus(doc_id,text[,source_id,start,end]); queries(query_id,text[,scope]);
    qrels(query_id,doc_id,split,score); candidates(query_id,doc_id);
    split_documents(split,doc_id) for independent code retrieval candidate pools;
    spans(query_id,source_id,start,end); esci_labels(query_id,doc_id,label).
    """
    path = Path(root)
    bundle = RetrievalBundle(meta={"dataset": f"custom:{path}", "protocol": protocol})
    for record in _table(path, "corpus"):
        did = str(record["doc_id"])
        text = clean_text(record["text"])
        if not text:
            raise ValueError(f"Custom corpus doc {did!r} has empty text; repair source data")
        bundle.corpus[did] = text
        source = record.get("source_id")
        if source is not None and clean_text(source):
            scope = str(source)
            bundle.source_docs[did] = scope
            bundle.scope_docs.setdefault(scope, []).append(did)
            if record.get("start") is not None and record.get("end") is not None:
                bundle.offsets[did] = (int(record["start"]), int(record["end"]))
    for record in _table(path, "queries"):
        qid = str(record["query_id"])
        text = clean_text(record["text"])
        if not text:
            raise ValueError(f"Empty custom query: {qid!r}")
        bundle.queries[qid] = text
        if record.get("scope") is not None and clean_text(record["scope"]):
            bundle.scopes[qid] = str(record["scope"])
    for record in _table(path, "qrels"):
        qid = str(record["query_id"])
        did = str(record["doc_id"])
        split = str(record["split"]).lower()
        if split in {"dev", "validation"}:
            split = "val"
        if split not in bundle.qrels:
            raise ValueError(f"Unsupported custom split {split}; use train/val/test")
        grade = float(record.get("score", record.get("relevance", record.get("grade", 0))))
        bundle.qrels[split].setdefault(qid, {})[did] = grade
    # For split-scoped retrieval, candidate membership is an independent data
    # source; inferring it from qrels would leak relevance ground truth.
    for record in _table(path, "split_documents", optional=True):
        split = str(record["split"]).lower()
        if split in {"dev", "validation"}:
            split = "val"
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split_documents split: {split!r}")
        bundle.split_docs.setdefault(split, []).append(str(record["doc_id"]))
    for record in _table(path, "candidates", optional=True):
        bundle.pools.setdefault(str(record["query_id"]), []).append(str(record["doc_id"]))
    for qid, pool in bundle.pools.items():
        bundle.pools[qid] = list(dict.fromkeys(pool))
    for record in _table(path, "spans", optional=True):
        bundle.gold_spans.setdefault(str(record["query_id"]), []).append(
            (str(record["source_id"]), int(record["start"]), int(record["end"]))
        )
    for record in _table(path, "esci_labels", optional=True):
        bundle.esci_labels[(str(record["query_id"]), str(record["doc_id"]))] = str(record["label"])
    if bundle.gold_spans and bundle.scopes:
        bundle.meta["protocol"] = "document_scoped"
    elif bundle.pools and protocol == "full_corpus":
        bundle.meta["protocol"] = "judged_candidates"
    bundle.meta["split_manifest"] = {qid: split for split, qrels in bundle.qrels.items() for qid in qrels}
    return bundle

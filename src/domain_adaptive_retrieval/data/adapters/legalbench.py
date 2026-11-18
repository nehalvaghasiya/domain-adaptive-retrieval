"""LegalBench-RAG evidence spans, source-grouped splits and exact offsets."""

import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.chunks import chunk_words
from domain_adaptive_retrieval.data.cleaning import clean_text
from domain_adaptive_retrieval.data.splits import grouped_splits

LEGAL_SOURCES = ("privacy_qa", "contractnli", "cuad", "maud")


def _corpus_path(root: Path, source: str) -> Path:
    """Resolve canonically equivalent filenames without changing text or source IDs."""
    expected = root / source
    if expected.is_file():
        return expected
    normalized = unicodedata.normalize("NFC", source)
    matches = [
        path
        for path in root.rglob("*.txt")
        if path.is_file()
        and unicodedata.normalize("NFC", path.relative_to(root).as_posix())
        == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous Unicode-equivalent legal corpus paths: {source!r}")
    raise FileNotFoundError(f"LegalBench corpus file missing: {expected}")


def load_legalbench(
    root: str | Path, *, seed: int = 42, width: int = 180, overlap: int = 40
) -> RetrievalBundle:
    """Create passage labels by overlapping original exact legal evidence spans."""
    path = Path(root)
    tests: list[tuple[str, int, dict[str, Any]]] = []
    for src in LEGAL_SOURCES:
        benchmark = path / "benchmarks" / f"{src}.json"
        if not benchmark.is_file():
            raise FileNotFoundError(
                f"LegalBench missing {benchmark}; download awinml/legalbench-rag"
            )
        tests.extend(
            (src, index, row)
            for index, row in enumerate(
                json.loads(benchmark.read_text(encoding="utf-8-sig"))["tests"]
            )
        )
    assignment: dict[str, str] = {}
    for source in LEGAL_SOURCES:
        paths = {
            item["file_path"]
            for src, _, row in tests
            if src == source
            for item in row["snippets"]
        }
        # Split inside each benchmark source. Many documents share a source.
        assignment.update(grouped_splits(paths, seed=seed))
    bundle = RetrievalBundle(
        meta={
            "dataset": "awinml/legalbench-rag",
            "protocol": "document_scoped",
            "split_manifest": assignment,
        }
    )
    original: dict[str, str] = {}
    by_source: dict[str, list[str]] = defaultdict(list)
    for source in sorted(assignment):
        document = _corpus_path(path / "corpus", source).read_text(encoding="utf-8")
        original[source] = document
        for index, start, end, content in chunk_words(document, width, overlap):
            did = f"{source}::c{index}"
            bundle.corpus[did] = content
            bundle.source_docs[did] = source
            bundle.offsets[did] = (start, end)
            by_source[source].append(did)
    bundle.scope_docs = dict(by_source)
    bundle.train_docs = {
        did
        for did, source in bundle.source_docs.items()
        if assignment[source] == "train"
    }
    for source, index, row in tests:
        qid = f"{source}:{index}"
        bundle.queries[qid] = clean_text(row["query"])
        spans: list[tuple[str, int, int]] = []
        for item in row["snippets"]:
            name = str(item["file_path"])
            start, end = map(int, item["span"])
            if original[name][start:end] != item["answer"]:
                raise ValueError(
                    f"Legal evidence exact offset mismatch: {qid} {name}:{start}:{end}"
                )
            spans.append((name, start, end))
        sources = {name for name, _, _ in spans}
        if len(sources) != 1:
            raise ValueError(
                f"Legal query {qid} spans multiple documents; unsupported scoped protocol"
            )
        origin = next(iter(sources))
        bundle.scopes[qid] = origin
        bundle.gold_spans[qid] = spans
        labels: dict[str, float] = {}
        for did in by_source[origin]:
            left, right = bundle.offsets[did]
            if any(left < end and right > start for _, start, end in spans):
                labels[did] = 1.0
        if not labels:
            raise ValueError(f"No indexed passage overlaps legal evidence: {qid}")
        bundle.qrels[assignment[origin]][qid] = labels
    return bundle


def download_legalbench(cache_dir: str | Path) -> Path:
    """Fetch original LegalBench-RAG dataset only when explicitly requested."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Download legal data: uv sync --extra train") from exc
    return Path(
        snapshot_download(
            repo_id="awinml/legalbench-rag",
            repo_type="dataset",
            local_dir=str(Path(cache_dir) / "legalbench-rag"),
        )
    )

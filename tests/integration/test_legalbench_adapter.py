"""Run real LegalBench adapter against source-structured local mini-benchmark."""

import json
from pathlib import Path

import pytest
from domain_adaptive_retrieval.data.adapters.legalbench import (
    LEGAL_SOURCES,
    load_legalbench,
)
from domain_adaptive_retrieval.data.validation import validate_bundle


def fixture_legalbench(root: Path) -> Path:
    (root / "benchmarks").mkdir(parents=True)
    (root / "corpus").mkdir(parents=True)
    for source in LEGAL_SOURCES:
        tests = []
        for index in range(3):
            relative = f"{source}/doc{index}.txt"
            doc_path = root / "corpus" / relative
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            document = f"The {source} clause states: exact evidence {index}. Extra legal context."
            doc_path.write_text(document, encoding="utf-8")
            start = document.index("exact evidence")
            end = start + len("exact evidence")
            tests.append(
                {
                    "query": f"Find evidence {index}",
                    "snippets": [
                        {
                            "file_path": relative,
                            "span": [start, end],
                            "answer": document[start:end],
                        }
                    ],
                }
            )
        (root / "benchmarks" / f"{source}.json").write_text(
            json.dumps({"tests": tests}), encoding="utf-8"
        )
    return root


def test_legalbench_local_adapter(tmp_path: Path) -> None:
    bundle = load_legalbench(fixture_legalbench(tmp_path), width=6, overlap=2)
    audit = validate_bundle(bundle)
    assert audit["protocol"] == "document_scoped"
    assert all(bundle.qrels[split] for split in ("train", "val", "test"))
    assert len(bundle.scopes) == 12
    for qid, scope in bundle.scopes.items():
        assert all(bundle.source_docs[did] == scope for did in bundle.scope_docs[scope])
        assert all(
            did in bundle.corpus
            for did in bundle.qrels[
                next(split for split in bundle.qrels if qid in bundle.qrels[split])
            ][qid]
        )


def test_legalbench_exact_offset_mismatch(tmp_path: Path) -> None:
    root = fixture_legalbench(tmp_path)
    path = root / "benchmarks" / "cuad.json"
    raw = json.loads(path.read_text())
    raw["tests"][0]["snippets"][0]["answer"] = "corrupt evidence"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exact offset mismatch"):
        load_legalbench(root)

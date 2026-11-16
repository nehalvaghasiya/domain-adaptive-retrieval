"""Reproduce the exact class of FiQA missing-document failure reported in Colab."""

import csv
import json
from pathlib import Path

import pytest

from domain_adaptive_retrieval.data.adapters.beir import load_beir
from domain_adaptive_retrieval.data.validation import validate_bundle


def _beir_fixture(root: Path, *, corrupt: bool = False) -> Path:
    dataset = root / "beir" / "fiqa"
    (dataset / "qrels").mkdir(parents=True)
    with (dataset / "corpus.jsonl").open("w") as file:
        for did, text in (
            ("doc0", "valid financial explanation"),
            ("docempty", ""),
            ("doc1", "another explanation"),
        ):
            file.write(json.dumps({"_id": did, "text": text, "title": ""}) + "\n")
    with (dataset / "queries.jsonl").open("w") as file:
        for qid in ("461", "462", "463"):
            file.write(json.dumps({"_id": qid, "text": "What is a dividend?"}) + "\n")
    for filename, qid in (("train", "461"), ("dev", "462"), ("test", "463")):
        with (dataset / "qrels" / f"{filename}.tsv").open("w", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(["query-id", "corpus-id", "score"])
            writer.writerow(
                [qid, "ghost" if corrupt and filename == "test" else "doc0", "1"]
            )
            writer.writerow([qid, "docempty", "1"])
    return root


def test_filtered_empty_doc_does_not_leave_dangling_qrel(tmp_path: Path) -> None:
    loaded = load_beir("finance", _beir_fixture(tmp_path))
    audit = validate_bundle(loaded)
    assert "docempty" not in loaded.corpus
    assert loaded.qrels["train"]["461"] == {"doc0": 1.0}
    assert audit["data_quality"]["filtered_empty_documents"] == 1
    assert audit["data_quality"]["filtered_qrels"]["train"] == 1


def test_truly_missing_beir_doc_is_hard_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="truly missing doc"):
        load_beir("finance", _beir_fixture(tmp_path, corrupt=True))

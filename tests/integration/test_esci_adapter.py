"""Amazon ESCI Parquet query/product stream is tested without gigabyte downloads."""

import sys
import types
from pathlib import Path

import pandas as pd

from domain_adaptive_retrieval.data.adapters.esci import load_esci
from domain_adaptive_retrieval.data.validation import validate_bundle


class FakeColumn:
    def __init__(self, records):
        self.records = records

    def to_pylist(self):
        return self.records


class FakeBatch:
    def __init__(self, records):
        self.columns = [
            FakeColumn([row[index] for row in records]) for index in range(3)
        ]


class FakeParquetFile:
    def __init__(self, _path):
        pass

    def iter_batches(self, **_options):
        yield FakeBatch([(f"p{i}", f"Product title {i}", "us") for i in range(16)])


def test_esci_label_and_pool_protocol(monkeypatch, tmp_path: Path) -> None:
    for name in (
        "shopping_queries_dataset_examples.parquet",
        "shopping_queries_dataset_products.parquet",
    ):
        (tmp_path / name).write_bytes(b"local stub")
    examples = pd.DataFrame(
        [
            {
                "query_id": qid,
                "product_id": f"p{qid * 4 + j}",
                "query": f"Find item {qid}",
                "product_locale": "us",
                "esci_label": label,
                "small_version": 1,
                "split": "train" if qid < 3 else "test",
            }
            for qid in range(4)
            for j, label in enumerate("ESCI")
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: examples)
    module = types.ModuleType("pyarrow.parquet")
    module.ParquetFile = FakeParquetFile
    try:
        import pyarrow
    except ImportError:
        pyarrow = types.ModuleType("pyarrow")
        monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setattr(pyarrow, "parquet", module, raising=False)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", module)
    bundle = load_esci(tmp_path, seed=42)
    audit = validate_bundle(bundle)
    assert audit["protocol"] == "judged_candidates"
    assert sum(len(split) for split in bundle.qrels.values()) == 4
    assert all(len(pool) == 4 for pool in bundle.pools.values())
    assert set(bundle.esci_labels.values()) == {"E", "S", "C", "I"}
    assert all(
        bundle.qrels[split][qid]["us:p0"] == 3
        for split in bundle.qrels
        for qid in bundle.qrels[split]
        if qid == "esci:0"
    )

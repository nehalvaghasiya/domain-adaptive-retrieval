"""CodeSearchNet official split streaming without network dependencies."""

import sys
import types

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.adapters.codesearchnet import load_codesearchnet
from domain_adaptive_retrieval.data.validation import validate_bundle


class FakeStream:
    def __init__(self, split: str):
        self.split = split

    def shuffle(self, **_options: object) -> "FakeStream":
        return self

    def __iter__(self):
        for i in range(3):
            yield {
                "func_documentation_string": f"Calculate a distinct numerical quantity for {self.split} task number {i}",
                "func_code_string": f'def task_{self.split}_{i}(x):\n    """Calculate a distinct numerical quantity for {self.split} task number {i}"""\n    output = x + {i}\n    return output * (x - 2)\n',
            }


def test_codesearchnet_streaming(monkeypatch) -> None:
    module = types.ModuleType("datasets")
    module.load_dataset = lambda repo, config, split, streaming: FakeStream(split)
    monkeypatch.setitem(sys.modules, "datasets", module)
    cfg = ExperimentConfig(
        domain="code", code_max_train=2, code_max_val=2, code_max_test=2
    )
    bundle = load_codesearchnet(cfg)
    audit = validate_bundle(bundle)
    assert audit["split_queries"] == {"train": 2, "val": 2, "test": 2}
    assert bundle.meta["capped_run"]
    assert all('"""' not in text for text in bundle.corpus.values())
    assert len(bundle.train_docs) == 2


def test_codesearchnet_candidates_use_declared_split_not_gold(monkeypatch) -> None:
    """Do not peek at qrels to construct the allowed retrieval pool."""
    from domain_adaptive_retrieval.retrieval.candidates import allowed_ids

    module = types.ModuleType("datasets")
    module.load_dataset = lambda repo, config, split, streaming: FakeStream(split)
    monkeypatch.setitem(sys.modules, "datasets", module)
    bundle = load_codesearchnet(
        ExperimentConfig(
            domain="code", code_max_train=3, code_max_val=3, code_max_test=3
        )
    )
    qid = next(iter(bundle.qrels["test"]))
    original = list(allowed_ids(bundle, qid, "test"))
    assert len(original) == 3
    # One document may be not judged for a given query, but remains a candidate.
    bundle.qrels["test"][qid] = {next(iter(bundle.qrels["test"][qid])): 1.0}
    assert allowed_ids(bundle, qid, "test") == original
    validate_bundle(bundle)


def test_codesearchnet_missing_split_membership_fails(monkeypatch) -> None:
    import pytest
    from domain_adaptive_retrieval.data.validation import DatasetIntegrityError
    from domain_adaptive_retrieval.retrieval.candidates import allowed_ids

    module = types.ModuleType("datasets")
    module.load_dataset = lambda repo, config, split, streaming: FakeStream(split)
    monkeypatch.setitem(sys.modules, "datasets", module)
    bundle = load_codesearchnet(
        ExperimentConfig(
            domain="code", code_max_train=2, code_max_val=2, code_max_test=2
        )
    )
    bundle.split_docs.pop("test")
    with pytest.raises(DatasetIntegrityError, match="membership missing"):
        validate_bundle(bundle)
    with pytest.raises(ValueError, match="not declared"):
        allowed_ids(bundle, bundle.split_queries("test")[0], "test")

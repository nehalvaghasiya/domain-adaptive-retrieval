"""Exact CPU Torch chunked retrieval, ties, shape and invalid dimensions."""

import numpy as np
import pytest
from domain_adaptive_retrieval.retrieval.cuda_topk import chunked_topk


@pytest.mark.parametrize(
    "qbatch,dchunk,k", [(1, 1, 1), (3, 2, 4), (4, 7, 10), (2, 50, 17)]
)
def test_chunked_topk_matches_numpy_full(qbatch: int, dchunk: int, k: int) -> None:
    rng = np.random.default_rng(77)
    queries = rng.standard_normal((7, 17)).astype(np.float32)
    docs = rng.standard_normal((31, 17)).astype(np.float32)
    ids = [f"doc-{i:03}" for i in range(len(docs))]
    out = chunked_topk(
        queries, docs, ids, top_k=k, query_batch=qbatch, doc_chunk=dchunk
    )
    expected = queries @ docs.T
    for row, ranked in enumerate(out):
        top = sorted(
            zip(ids, expected[row], strict=True),
            key=lambda record: (-record[1], record[0]),
        )[:k]
        assert [did for did, _ in ranked] == [did for did, _ in top]
        assert np.allclose(
            [score for _, score in ranked],
            [float(score) for _, score in top],
            atol=1e-4,
        )


def test_topk_ties_and_out_of_order_ids() -> None:
    queries = np.array([[1.0, 0]], dtype=np.float32)
    docs = np.array([[1.0, 0], [1.0, 0], [0.0, 1]], dtype=np.float32)
    ranked = chunked_topk(queries, docs, ["z", "a", "c"], top_k=2, doc_chunk=1)
    assert [did for did, _ in ranked[0]] == ["a", "z"]


def test_topk_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="share dimension"):
        chunked_topk(np.zeros((1, 3)), np.zeros((2, 4)), ["a", "b"])


def test_topk_ids_length_mismatch() -> None:
    with pytest.raises(ValueError, match="doc_ids length"):
        chunked_topk(np.zeros((1, 3)), np.zeros((2, 3)), ["a"])

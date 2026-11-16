"""Exact CUDA ranking equivalence on genuine GPU hardware."""

import numpy as np
import pytest
import torch

from domain_adaptive_retrieval.retrieval.cuda_topk import chunked_topk


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires T4 or equivalent")
def test_chunked_topk_on_cuda_matches_cpu() -> None:
    rng = np.random.default_rng(10)
    q = rng.random((11, 31), dtype=np.float32)
    d = rng.random((103, 31), dtype=np.float32)
    ids = [f"doc{i:03}" for i in range(103)]
    cpu = chunked_topk(q, d, ids, device="cpu", doc_chunk=19, query_batch=3)
    gpu = chunked_topk(q, d, ids, device="cuda:0", doc_chunk=19, query_batch=3)
    for a, b in zip(cpu, gpu, strict=True):
        assert [did for did, _ in a] == [did for did, _ in b]

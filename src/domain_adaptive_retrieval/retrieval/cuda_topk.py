"""Exact chunked GPU/CPU cosine top-K with deterministic tie handling."""

from typing import Any

import numpy as np


def chunked_topk(
    query_vectors: np.ndarray,
    doc_vectors: np.ndarray,
    doc_ids: list[str],
    *,
    top_k: int = 100,
    device: str = "cpu",
    query_batch: int = 16,
    doc_chunk: int = 8192,
) -> list[list[tuple[str, float]]]:
    """Compute normalized dot-product rankings, never materializing the full matrix.

    Returned results break equal score ties by document ID; all matrix operations
    happen on the selected Torch device when CUDA is requested.
    """
    if (
        query_vectors.ndim != 2
        or doc_vectors.ndim != 2
        or query_vectors.shape[1] != doc_vectors.shape[1]
    ):
        raise ValueError(
            "Query and document vectors must be rank-2 and share dimension"
        )
    if doc_vectors.shape[0] != len(doc_ids):
        raise ValueError("doc_ids length != number of document vectors")
    if top_k < 1 or query_batch < 1 or doc_chunk < 1:
        raise ValueError("top_k/query_batch/doc_chunk must be positive")
    if not doc_ids:
        return [[] for _ in range(len(query_vectors))]
    order = sorted(range(len(doc_ids)), key=lambda i: doc_ids[i])
    sorted_ids = [doc_ids[i] for i in order]
    sorted_vectors = np.asarray(doc_vectors[order], dtype=np.float32)
    qv = np.asarray(query_vectors, dtype=np.float32)
    k = min(top_k, len(sorted_ids))
    if device.startswith("cuda"):
        from domain_adaptive_retrieval.models.device import check_device

        check_device(device, require_gpu=True)
    try:
        import torch
    except ImportError:
        if device != "cpu":
            raise RuntimeError("CUDA top-K requires torch") from None
        return _numpy_topk(qv, sorted_vectors, sorted_ids, k, query_batch, doc_chunk)

    all_rankings: list[list[tuple[str, float]]] = []
    for qstart in range(0, len(qv), query_batch):
        queries = torch.as_tensor(
            qv[qstart : qstart + query_batch], dtype=torch.float32, device=device
        )
        best_scores: Any = torch.empty(
            (queries.shape[0], 0), dtype=torch.float32, device=device
        )
        best_indices: Any = torch.empty(
            (queries.shape[0], 0), dtype=torch.long, device=device
        )
        with torch.inference_mode():
            for first in range(0, len(sorted_ids), doc_chunk):
                docs = torch.as_tensor(
                    sorted_vectors[first : first + doc_chunk],
                    dtype=torch.float32,
                    device=device,
                )
                scores = queries @ docs.T
                idx = torch.arange(
                    first, first + docs.shape[0], dtype=torch.long, device=device
                ).expand(len(queries), -1)
                merged_scores = torch.cat((best_scores, scores), dim=1)
                merged_idx = torch.cat((best_indices, idx), dim=1)
                # Earlier chunks have smaller document indices. Stable sorting
                # keeps their priority when floating scores tie exactly.
                stable = torch.argsort(
                    merged_scores, dim=1, descending=True, stable=True
                )[:, :k]
                best_scores = torch.gather(merged_scores, 1, stable)
                best_indices = torch.gather(merged_idx, 1, stable)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        for indices, scores in zip(
            best_indices.cpu().tolist(), best_scores.cpu().tolist(), strict=True
        ):
            all_rankings.append(
                [
                    (sorted_ids[index], float(score))
                    for index, score in zip(indices, scores, strict=True)
                ]
            )
    return all_rankings


def _numpy_topk(
    queries: np.ndarray,
    docs: np.ndarray,
    ids: list[str],
    k: int,
    qbatch: int,
    dchunk: int,
) -> list[list[tuple[str, float]]]:
    """Chunked NumPy fallback for lightweight CPU-only installations."""
    result: list[list[tuple[str, float]]] = []
    for qstart in range(0, len(queries), qbatch):
        qv = queries[qstart : qstart + qbatch]
        best: list[list[tuple[int, float]]] = [[] for _ in range(len(qv))]
        for first in range(0, len(ids), dchunk):
            scores = qv @ docs[first : first + dchunk].T
            for row in range(len(qv)):
                merged = best[row] + [
                    (first + i, float(score)) for i, score in enumerate(scores[row])
                ]
                best[row] = sorted(merged, key=lambda pair: (-pair[1], pair[0]))[:k]
        for ranking in best:
            result.append([(ids[i], score) for i, score in ranking])
    return result

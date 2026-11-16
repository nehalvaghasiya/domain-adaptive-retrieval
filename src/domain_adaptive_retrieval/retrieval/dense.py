"""Dense retrieval scorer that works with a real embedder or a lightweight fake."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.retrieval.candidates import Predictions, allowed_ids
from domain_adaptive_retrieval.retrieval.cuda_topk import chunked_topk


class DenseRetriever:
    """Encode once, search globally or over query-specific candidate scopes."""

    def __init__(
        self,
        model: Any,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "",
        query_batch: int = 16,
        doc_chunk: int = 8192,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.query_batch = query_batch
        self.doc_chunk = doc_chunk

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(values, dtype=np.float32)

    def search(
        self,
        bundle: RetrievalBundle,
        qids: Sequence[str],
        *,
        split: str,
        top_k: int = 100,
    ) -> Predictions:
        """Search without using gold labels except to define an official held-out corpus pool."""
        query_ids = list(qids)
        if not query_ids:
            return {}
        doc_ids = sorted(
            {did for qid in query_ids for did in allowed_ids(bundle, qid, split)}
        )
        if not doc_ids:
            raise ValueError("No searchable documents for selected query scope")
        docs = self._encode([bundle.corpus[did] for did in doc_ids])
        questions = self._encode(
            [self.query_prefix + bundle.queries[qid] for qid in query_ids]
        )
        if len(questions.shape) != 2 or len(docs.shape) != 2:
            raise ValueError("Embedding model must return 2D float vectors")
        scoped = bool(bundle.scopes or bundle.pools)
        if not scoped:
            rankings = chunked_topk(
                questions,
                docs,
                doc_ids,
                top_k=top_k,
                device=self.device,
                query_batch=self.query_batch,
                doc_chunk=self.doc_chunk,
            )
            return dict(zip(query_ids, rankings, strict=True))
        locations = {did: i for i, did in enumerate(doc_ids)}
        results: Predictions = {}
        for i, qid in enumerate(query_ids):
            selected = sorted(set(allowed_ids(bundle, qid, split)))
            positions = [locations[did] for did in selected]
            rank = chunked_topk(
                questions[i : i + 1],
                docs[positions],
                selected,
                top_k=top_k,
                device=self.device,
                query_batch=self.query_batch,
                doc_chunk=self.doc_chunk,
            )
            results[qid] = rank[0]
        return results

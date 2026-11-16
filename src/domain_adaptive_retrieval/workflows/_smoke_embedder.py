"""Deterministic synthetic-only embedder used by offline integration smoke."""

import hashlib

import numpy as np

from domain_adaptive_retrieval.retrieval.bm25 import tokenize


class HashingEmbedder:
    """Fake deterministic text embedder; not trained and never used in real benchmarks."""

    def __init__(self, *, dimensions: int = 64, salt: str = "base") -> None:
        self.dimensions = dimensions
        self.salt = salt

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        """Produce normalized token-hash vectors for pipeline tests only."""
        rows = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in tokenize(text):
                signature = int.from_bytes(
                    hashlib.sha256((self.salt + word).encode()).digest()[:8], "big"
                )
                rows[i, signature % self.dimensions] += 1.0
            norm = np.linalg.norm(rows[i])
            if norm:
                rows[i] /= norm
        return rows

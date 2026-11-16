"""Deterministic synthetic-only relevance scorer used by offline smoke."""

import numpy as np

from domain_adaptive_retrieval.retrieval.bm25 import tokenize


class LexicalReranker:
    """Fake word-overlap scoring; not a trained cross-encoder."""

    def __init__(self, *, adjusted: bool = False) -> None:
        self.adjusted = adjusted

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> np.ndarray:
        """Return deterministic lexical-overlap scores for integration testing."""
        scores: list[float] = []
        for query, text in pairs:
            words = set(tokenize(query))
            document = set(tokenize(text))
            score = len(words & document) / max(len(words), 1)
            if self.adjusted:
                score += 0.001 * len(document)
            scores.append(score)
        return np.asarray(scores, dtype=np.float32)

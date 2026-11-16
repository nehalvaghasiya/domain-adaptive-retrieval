"""Tiny dependency-free BM25 lexical baseline with deterministically ordered results."""

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens; retain numbers for codes/products."""
    return re.findall(r"\w+", text.casefold())


class BM25Retriever:
    """Corpus-indexed Okapi BM25; default `k1=1.5, b=0.75`."""

    def __init__(
        self, corpus: Mapping[str, str], k1: float = 1.5, b: float = 0.75
    ) -> None:
        if not corpus:
            raise ValueError("BM25 corpus must not be empty")
        self.docs = sorted(corpus)
        self.freq = {did: Counter(tokenize(corpus[did])) for did in self.docs}
        self.length = {did: sum(freq.values()) for did, freq in self.freq.items()}
        self.avg_len = sum(self.length.values()) / len(self.docs)
        self.k1, self.b = k1, b
        counts: Counter[str] = Counter()
        for tokens in self.freq.values():
            counts.update(tokens.keys())
        self.idf = {
            term: math.log1p((len(self.docs) - n + 0.5) / (n + 0.5))
            for term, n in counts.items()
        }

    def search(
        self, query: str, *, top_k: int = 100, allowed: Sequence[str] | None = None
    ) -> list[tuple[str, float]]:
        """Rank by BM25, stable ties broken by lexicographic document ID."""
        if top_k <= 0:
            return []
        query_counts = Counter(tokenize(query))
        ids = list(allowed) if allowed is not None else self.docs
        result: list[tuple[str, float]] = []
        for did in ids:
            tf = self.freq[did]
            length = self.length[did]
            norm = self.k1 * (1 - self.b + self.b * length / max(self.avg_len, 1))
            score = sum(
                weight
                * self.idf.get(term, 0.0)
                * (count * (self.k1 + 1) / (count + norm))
                for term, weight in query_counts.items()
                if (count := tf.get(term, 0)) > 0
            )
            result.append((did, score))
        return sorted(result, key=lambda record: (-record[1], record[0]))[:top_k]

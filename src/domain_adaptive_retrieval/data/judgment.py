"""Relevance judgment value object."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RelevanceJudgment:
    """A query/document relevance grade in a named split."""

    query_id: str
    doc_id: str
    grade: float
    split: str

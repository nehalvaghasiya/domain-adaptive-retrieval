"""Explicit task-dependent evaluation protocols."""

from enum import StrEnum


class EvaluationProtocol(StrEnum):
    """Controls search scope and admissible metrics."""

    FULL_CORPUS = "full_corpus"
    JUDGED_CANDIDATES = "judged_candidates"
    DOCUMENT_SCOPED = "document_scoped"
    HELD_OUT_CORPUS = "held_out_corpus"

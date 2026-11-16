"""Full matrix must retain frozen candidate membership across reranking."""

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.evaluation.matrix import run_matrix
from domain_adaptive_retrieval.workflows._smoke_embedder import HashingEmbedder
from domain_adaptive_retrieval.workflows._smoke_reranker import LexicalReranker


def test_matrix_seven_systems(finance_bundle: RetrievalBundle) -> None:
    matrix = run_matrix(
        finance_bundle,
        base_embedder=HashingEmbedder(salt="base"),
        tuned_embedder=HashingEmbedder(salt="adapted"),
        base_reranker=LexicalReranker(),
        tuned_reranker=LexicalReranker(adjusted=True),
        top_k=5,
    )
    assert list(matrix["metrics"]) == [f"S{i}" for i in range(7)]
    for left, right in (("S1", "S3"), ("S1", "S4"), ("S2", "S5"), ("S2", "S6")):
        for qid in matrix["predictions"][left]:
            source = {did for did, _ in matrix["predictions"][left][qid]}
            target = {did for did, _ in matrix["predictions"][right][qid]}
            assert source == target

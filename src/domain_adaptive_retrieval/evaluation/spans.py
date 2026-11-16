"""Source-aware, overlap-safe legal character span evaluation."""

from collections import defaultdict
from collections.abc import Iterable, Sequence

from domain_adaptive_retrieval.data.bundle import RetrievalBundle


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union overlapping/touching half-open intervals without double counting."""
    result: list[list[int]] = []
    for left, right in sorted(intervals):
        if left < 0 or right < left:
            raise ValueError("Invalid character offset")
        if not result or left > result[-1][1]:
            result.append([left, right])
        else:
            result[-1][1] = max(result[-1][1], right)
    return [(left, right) for left, right in result]


def _length(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(right - left for left, right in merge_intervals(intervals))


def span_metrics(
    bundle: RetrievalBundle, query_id: str, ranked_ids: Sequence[str], k: int = 5
) -> dict[str, float]:
    """Character-level precision and recall within each original source document."""
    gold: dict[str, list[tuple[int, int]]] = defaultdict(list)
    predicted: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for source, left, right in bundle.gold_spans.get(query_id, []):
        gold[source].append((left, right))
    for did in ranked_ids[:k]:
        if did in bundle.offsets:
            predicted[bundle.source_docs[did]].append(bundle.offsets[did])
    gold_total = selected_total = matched = 0
    for source in set(gold) | set(predicted):
        a = merge_intervals(gold[source])
        b = merge_intervals(predicted[source])
        gold_total += _length(a)
        selected_total += _length(b)
        for left1, right1 in a:
            for left2, right2 in b:
                matched += max(0, min(right1, right2) - max(left1, left2))
    return {
        f"legal_character_recall@{k}": matched / gold_total if gold_total else 0.0,
        f"legal_character_precision@{k}": matched / selected_total
        if selected_total
        else 0.0,
    }

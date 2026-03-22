from __future__ import annotations

from collections import Counter

from src.retrieval import RetrievalMatch
from src.utils import DrawRecord


def _context_summary(context: list[DrawRecord]) -> dict:
    freq: Counter[int] = Counter()
    for r in context:
        freq.update(r.numbers)
    return {
        "context_rows": len(context),
        "top_hot_numbers": [int(n) for n, _ in freq.most_common(10)],
        "top_cold_numbers": [int(n) for n, _ in sorted(freq.items(), key=lambda x: x[1])[:10]],
    }


def build_prediction_explain(
    context: list[DrawRecord],
    top20: list[int],
    top10: list[int],
    top3: list[int],
    matches: list[RetrievalMatch],
) -> dict:
    return {
        "top20": top20,
        "top10": top10,
        "top3": top3,
        "context_summary": _context_summary(context),
        "retrieval_summary": [
            {
                "end_issue": m.end_issue,
                "similarity": m.similarity,
                "exact_draw_match_count": m.exact_draw_match_count,
                "same_day_progress": m.same_day_progress,
            }
            for m in matches[:5]
        ],
    }

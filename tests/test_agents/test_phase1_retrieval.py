from datetime import date, timedelta

from src.retrieval import SimilarWindowRetriever
from src.utils import DrawRecord


def _record(i: int) -> DrawRecord:
    nums = tuple(range((i % 50) + 1, (i % 50) + 21))
    return DrawRecord(issue=str(i), draw_date=date(2026, 1, 1) + timedelta(days=i // 20), numbers=nums, day_issue_index=(i % 20) + 1)


def test_retrieval_window_length() -> None:
    history = [_record(i) for i in range(200)]
    retriever = SimilarWindowRetriever(window_size=30, top_k=7)
    matches = retriever.query(history[:-20], history[-50:-20], day_issue_index=10)
    assert len(matches) <= 7

from src.retrieval import SimilarWindowRetriever


def test_retrieval_vectorized_matches_legacy(synthetic_records) -> None:
    history = synthetic_records[:-1]
    target_window = history[-30:]
    retriever = SimilarWindowRetriever(top_k=12, coarse_multiplier=1000)

    legacy = retriever._query_legacy_python(history=history, target_window=target_window, day_issue_index=target_window[-1].day_issue_index)
    current = retriever.query(history=history, target_window=target_window, day_issue_index=target_window[-1].day_issue_index)

    assert len(legacy) == len(current)
    for a, b in zip(legacy, current):
        assert a.end_issue == b.end_issue
        assert abs(a.similarity - b.similarity) <= 1e-12
        assert a.next_draw_numbers == b.next_draw_numbers
        assert a.same_day_progress == b.same_day_progress
        assert a.exact_draw_match_count == b.exact_draw_match_count
        assert a.exact_window_match == b.exact_window_match

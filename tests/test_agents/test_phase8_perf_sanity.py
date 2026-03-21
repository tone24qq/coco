from src.build_features import build_candidate_rows
from src.retrieval import SimilarWindowRetriever


def test_perf_sanity_retrieval_and_candidate_rows(synthetic_records) -> None:
    history = synthetic_records[:-1]
    context = history[-30:]
    retriever = SimilarWindowRetriever(top_k=20)
    matches = retriever.query(history=history, target_window=context, day_issue_index=context[-1].day_issue_index)
    assert len(matches) <= 20

    rows, _ = build_candidate_rows(
        history=history,
        issue=history[-1].issue,
        draw_date=history[-1].draw_date.isoformat(),
        label_numbers=set(synthetic_records[-1].numbers),
        min_dynamic_n=20,
        max_dynamic_n=40,
        top_k=20,
    )
    assert len(rows) == 80

import json

from src.build_features import (
    _build_candidate_rows_legacy_python,
    _history_profiles,
    _regime_transition,
    build_candidate_rows,
    resolve_dynamic_context,
)
from src.retrieval import RetrievalWeights, SimilarWindowRetriever


def test_build_candidate_rows_vectorized_equivalence(synthetic_records) -> None:
    history = synthetic_records[:150]
    issue = history[-1].issue
    draw_date = history[-1].draw_date.isoformat()
    label_numbers = set(synthetic_records[150].numbers)

    rows_new, matches_new = build_candidate_rows(
        history=history,
        issue=issue,
        draw_date=draw_date,
        label_numbers=label_numbers,
        min_dynamic_n=20,
        max_dynamic_n=40,
        top_k=8,
    )
    context = resolve_dynamic_context(history, min_dynamic_n=20, max_dynamic_n=40)
    retriever = SimilarWindowRetriever(top_k=8, weights=RetrievalWeights(), require_same_length_window=True, prefer_same_day_progress=True)
    matches_old = retriever._query_legacy_python(history=history, target_window=context, day_issue_index=context[-1].day_issue_index)
    rows_old = _build_candidate_rows_legacy_python(
        history=history,
        issue=issue,
        draw_date=draw_date,
        label_numbers=label_numbers,
        dynamic_n=len(context),
        matches=matches_old,
        prof_10=_history_profiles(history, min(10, len(history))),
        prof_20=_history_profiles(history, min(20, len(history))),
        prof_n=_history_profiles(history, min(len(history), max(20, len(context)))),
        transition=_regime_transition(history),
    )

    assert len(matches_new) == 8
    assert len(matches_old) == 8
    assert len(rows_new) == 80 == len(rows_old)
    for a, b in zip(rows_new, rows_old):
        assert set(a.keys()) == set(b.keys())
        for k in a:
            if isinstance(a[k], float):
                assert abs(float(a[k]) - float(b[k])) <= 1e-9, f"{k}"
            else:
                assert a[k] == b[k], f"{k}"

    # profile json compatibility
    assert json.loads(rows_new[0]["current_day_recent_n_profile"]) == json.loads(rows_old[0]["current_day_recent_n_profile"])

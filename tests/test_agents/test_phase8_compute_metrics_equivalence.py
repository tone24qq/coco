from src.modeling import _compute_metrics_legacy, compute_metrics, load_ranking_dataset


def test_compute_metrics_vectorized_equivalence(ranking_dataset_path) -> None:
    df = load_ranking_dataset(ranking_dataset_path)
    scored = df[["issue", "candidate_number", "label"]].copy()
    scored["final_score"] = (100.0 - scored["candidate_number"]).astype(float) + (scored["issue"].astype(str).str[-2:].astype(int) / 1000.0)
    legacy = _compute_metrics_legacy(scored)
    current = compute_metrics(scored)
    assert set(legacy.keys()) == set(current.keys())
    for k in legacy:
        assert abs(float(legacy[k]) - float(current[k])) <= 1e-12, k

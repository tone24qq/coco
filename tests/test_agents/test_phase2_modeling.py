from src.modeling import compute_metrics, load_ranking_dataset, make_time_series_splits, resolve_feature_columns


def test_time_series_split_is_ordered_and_non_leaky() -> None:
    issues = [f"I{i:03d}" for i in range(120)]
    splits = make_time_series_splits(issues, n_splits=3, min_train_issues=30)
    for train_issues, val_issues in splits:
        assert train_issues[-1] < val_issues[0]
        assert set(train_issues).isdisjoint(set(val_issues))


def test_modeling_contract_and_metrics(ranking_dataset_path) -> None:
    df = load_ranking_dataset(ranking_dataset_path)
    cols = resolve_feature_columns(df)
    assert cols
    scored = df[["issue", "candidate_number", "label"]].copy()
    scored["final_score"] = scored["candidate_number"].apply(lambda x: 100 - x)
    metrics = compute_metrics(scored)
    assert "top3_hit_rate" in metrics

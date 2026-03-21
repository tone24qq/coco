import numpy as np
import pandas as pd

from src.build_features import build_feature_rows
from src.modeling import _oof_ranker_scores, make_time_series_splits, resolve_feature_columns, run_cv
from src.ranking_dataset import attach_group_ids
from src.runtime_scoring import RuntimeWeights


def test_oof_and_cv_structure_equivalence(synthetic_records) -> None:
    rows = build_feature_rows(synthetic_records, min_history=60, retrieval_window=40, top_k=8)
    df = pd.DataFrame(attach_group_ids(rows))
    cols = resolve_feature_columns(df)

    splits = make_time_series_splits(df["issue"].tolist(), n_splits=3, min_train_issues=30)
    assert len(splits) >= 1
    for tr, va in splits:
        assert tr[-1] < va[0]

    oof = _oof_ranker_scores(df, cols)
    assert len(oof) == len(df)
    assert int(oof.notna().sum()) > 0
    assert int(np.isfinite(oof.dropna()).sum()) == int(oof.notna().sum())

    folds = run_cv(df, cols, RuntimeWeights.from_mapping({}), n_splits=3, min_train_issues=30)
    assert len(folds) == len(splits)
    for fold, (tr, va) in zip(folds, splits):
        assert fold.train_issues == tr
        assert fold.val_issues == va
        assert not fold.train_scored.empty
        assert not fold.val_scored.empty

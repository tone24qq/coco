import json
from pathlib import Path

import pandas as pd

from src.artifacts import load_cascade_artifacts, save_cascade_artifacts
from src.pipeline import CascadePipeline
from src.utils import build_issue_features


def _make_draws(n_rows: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        nums = [((i * 5 + k) % 80) + 1 for k in range(20)]
        rows.append(
            {
                "issue": 9000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(sorted(nums)),
            }
        )
    return pd.DataFrame(rows)


def test_cascade_artifacts_save_and_load(tmp_path: Path) -> None:
    feat_df = (
        build_issue_features(_make_draws(), min_history=22)
        .tail(30)
        .reset_index(drop=True)
    )
    params = {
        "iterations": 8,
        "learning_rate": 0.1,
        "depth": 4,
        "loss_function": "Logloss",
        "verbose": False,
        "random_seed": 42,
    }
    pipeline, artifacts = CascadePipeline.train(feat_df, 30, 10, params)
    _ = pipeline

    save_cascade_artifacts(
        tmp_path,
        artifacts,
        feature_version="v3_core20",
        train_issue_start=int(feat_df["issue"].min()),
        train_issue_end=int(feat_df["target_issue"].max()),
    )

    loaded = load_cascade_artifacts(tmp_path)
    loaded_pipeline = CascadePipeline.from_artifacts(loaded)
    out = loaded_pipeline.predict_issue(feat_df.iloc[-1])
    assert out["stage3_inputs"].shape[0] == 10
    assert out["final_scores"].shape == (80,)

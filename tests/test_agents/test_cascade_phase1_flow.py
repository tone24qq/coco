import json

import numpy as np
import pandas as pd

from src.pipeline import CascadePipeline
from src.utils import (
    CASCADE_V1_STAGE1_COLUMNS,
    CASCADE_V1_STAGE2_COLUMNS,
    CASCADE_V1_STAGE3_COLUMNS,
    build_issue_features,
    build_stage1_candidate_matrix,
    build_stage2_candidate_matrix,
    build_stage3_selector_inputs,
    get_pipeline_feature_schema,
    normalize_pipeline_version,
)


def _make_draws(n_rows: int = 120) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        nums = [((i * 3 + k) % 80) + 1 for k in range(20)]
        rows.append(
            {
                "issue": 7000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(sorted(nums)),
            }
        )
    return pd.DataFrame(rows)


def test_pipeline_schema_and_normalizer() -> None:
    assert normalize_pipeline_version("baseline_flat_score") == "baseline_flat_score"
    schema = get_pipeline_feature_schema("cascade_v1")
    assert list(schema["stage1"]) == CASCADE_V1_STAGE1_COLUMNS
    assert list(schema["stage2"]) == CASCADE_V1_STAGE2_COLUMNS
    assert list(schema["stage3"]) == CASCADE_V1_STAGE3_COLUMNS


def test_stage_builders_keep_contract_and_shapes() -> None:
    feat_df = build_issue_features(_make_draws(), min_history=22)
    issue_row = feat_df.iloc[-1]

    stage1 = build_stage1_candidate_matrix(issue_row, CASCADE_V1_STAGE1_COLUMNS)
    assert stage1.shape[0] == 80
    assert stage1.columns.tolist()[0] == "number"

    stage1 = stage1.copy()
    stage1["stage1_rank"] = np.arange(1, 81)
    stage1["stage1_keep_flag"] = (stage1["stage1_rank"] <= 30).astype(int)
    stage2 = build_stage2_candidate_matrix(issue_row, stage1, CASCADE_V1_STAGE2_COLUMNS)
    assert stage2["number"].nunique() == 30

    stage2 = stage2.copy()
    stage2["stage2_score"] = np.linspace(1.0, 0.0, len(stage2))
    stage2["stage2_rank"] = np.arange(1, len(stage2) + 1)
    stage2["stage2_keep_flag"] = (stage2["stage2_rank"] <= 10).astype(int)
    stage3 = build_stage3_selector_inputs(issue_row, stage2, top_k=10)
    assert stage3.columns.tolist() == CASCADE_V1_STAGE3_COLUMNS
    assert len(stage3) == 10


def test_cascade_pipeline_end_to_end_dataflow() -> None:
    feat_df = build_issue_features(_make_draws(), min_history=22)
    params = {
        "iterations": 10,
        "learning_rate": 0.1,
        "depth": 4,
        "loss_function": "Logloss",
        "verbose": False,
        "random_seed": 42,
    }
    pipeline, _ = CascadePipeline.train(feat_df.tail(40), 30, 10, params)
    out = pipeline.predict_issue(feat_df.iloc[-1])

    assert out["stage1"].shape[0] == 80
    assert int(out["stage1"]["stage1_keep_flag"].sum()) == 30
    assert int(out["stage2"]["stage2_keep_flag"].sum()) == 10
    assert out["stage3_inputs"].shape[0] == 10
    assert out["final_scores"].shape == (80,)

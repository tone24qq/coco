import json

import pandas as pd

from src.pipeline import CascadePipeline
from src.predict import resolve_runtime_strategy
from src.selector import build_selector_context, select_top3_combination
from src.utils import (
    CASCADE_V1_STAGE1_COLUMNS,
    CASCADE_V1_STAGE2_COLUMNS,
    CONFIG_DIR,
    build_issue_features,
    build_stage1_candidate_matrix,
    build_stage2_candidate_matrix,
    build_stage3_selector_inputs,
    load_yaml,
)


def _make_draws(n_rows: int = 90) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        nums = [((i * 7 + k) % 80) + 1 for k in range(20)]
        rows.append(
            {
                "issue": 11000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(sorted(nums)),
            }
        )
    return pd.DataFrame(rows)


def _train_pipeline() -> tuple[CascadePipeline, pd.DataFrame]:
    feat_df = (
        build_issue_features(_make_draws(), min_history=22)
        .tail(35)
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
    pipeline, _ = CascadePipeline.train(feat_df, 30, 10, params)
    return pipeline, feat_df


def test_selector_combination_output_and_reason() -> None:
    pipeline, feat_df = _train_pipeline()
    out = pipeline.predict_issue(feat_df.iloc[-1])
    assert len(out["final_top3"]) == 3
    assert len(set(out["final_top3"])) == 3
    assert isinstance(out["selector_reason"], str)
    assert out["selector_score"] == out["selector_score"]


def test_selector_only_uses_stage2_retained_top10() -> None:
    pipeline, feat_df = _train_pipeline()
    out = pipeline.predict_issue(feat_df.iloc[-1])
    stage2_keep = set(
        int(x)
        for x in out["stage2"][out["stage2"]["stage2_keep_flag"] == 1][
            "number"
        ].tolist()
    )
    stage3_numbers = set(int(x) for x in out["stage3_inputs"]["number"].tolist())
    assert stage3_numbers.issubset(stage2_keep)


def test_selector_context_does_not_need_target_label() -> None:
    pipeline, feat_df = _train_pipeline()
    issue_row = feat_df.iloc[-1].copy()
    issue_row_no_target = issue_row.drop(labels=["target_numbers", "target_issue"])
    out = pipeline.predict_issue(issue_row_no_target)
    ctx = build_selector_context(issue_row_no_target)
    sel = select_top3_combination(out["stage3_inputs"], ctx)
    assert len(sel.final_top3) == 3


def test_predict_issue_output_not_affected_by_target_numbers_column() -> None:
    pipeline, feat_df = _train_pipeline()
    row = feat_df.iloc[-1].copy()
    out_a = pipeline.predict_issue(row)

    row_mut = row.copy()
    row_mut["target_numbers"] = json.dumps([1, 2, 3])
    out_b = pipeline.predict_issue(row_mut)
    assert out_a["final_top3"] == out_b["final_top3"]


def test_stage3_schema_and_future_label_exclusion() -> None:
    pipeline, feat_df = _train_pipeline()
    out = pipeline.predict_issue(feat_df.iloc[-1])
    stage3 = out["stage3_inputs"]
    assert set(stage3.columns) == {
        "number",
        "zone_id",
        "tail",
        "stage2_score",
        "stage2_rank",
        "stage1_score",
    }
    for forbidden in ["target_numbers", "target_issue", "actual"]:
        assert forbidden not in stage3.columns


def test_stage1_temporal_boundary_invariant_to_future_labels() -> None:
    _, feat_df = _train_pipeline()
    row_a = feat_df.iloc[-1].copy()
    row_b = row_a.copy()
    row_b["target_numbers"] = json.dumps([1, 2, 3])
    row_b["target_issue"] = int(row_a["target_issue"]) + 999
    a = build_stage1_candidate_matrix(row_a, CASCADE_V1_STAGE1_COLUMNS)
    b = build_stage1_candidate_matrix(row_b, CASCADE_V1_STAGE1_COLUMNS)
    assert a.equals(b)


def test_stage2_temporal_boundary_invariant_to_future_labels() -> None:
    _, feat_df = _train_pipeline()
    row_a = feat_df.iloc[-1].copy()
    row_b = row_a.copy()
    row_b["target_numbers"] = json.dumps([4, 5, 6])
    row_b["target_issue"] = int(row_a["target_issue"]) + 123
    s1 = build_stage1_candidate_matrix(row_a, CASCADE_V1_STAGE1_COLUMNS)
    s1["stage1_rank"] = range(1, 81)
    s1["stage1_keep_flag"] = (s1["stage1_rank"] <= 30).astype(int)
    a = build_stage2_candidate_matrix(row_a, s1, CASCADE_V1_STAGE2_COLUMNS)
    b = build_stage2_candidate_matrix(row_b, s1, CASCADE_V1_STAGE2_COLUMNS)
    assert a.equals(b)


def test_stage3_temporal_boundary_invariant_to_future_labels() -> None:
    _, feat_df = _train_pipeline()
    row_a = feat_df.iloc[-1].copy()
    row_b = row_a.copy()
    row_b["target_numbers"] = json.dumps([7, 8, 9])
    row_b["target_issue"] = int(row_a["target_issue"]) + 42
    s1 = build_stage1_candidate_matrix(row_a, CASCADE_V1_STAGE1_COLUMNS)
    s1["stage1_rank"] = range(1, 81)
    s1["stage1_keep_flag"] = (s1["stage1_rank"] <= 30).astype(int)
    s2 = build_stage2_candidate_matrix(row_a, s1, CASCADE_V1_STAGE2_COLUMNS)
    s2["stage2_score"] = 1.0
    s2["stage2_rank"] = range(1, len(s2) + 1)
    s2["stage2_keep_flag"] = (s2["stage2_rank"] <= 10).astype(int)
    a = build_stage3_selector_inputs(row_a, s2, top_k=10)
    b = build_stage3_selector_inputs(row_b, s2, top_k=10)
    assert a.equals(b)


def test_predict_strategy_precedence_config_override() -> None:
    predict_cfg = {
        "pipeline": {
            "version": "cascade_v1",
            "artifact_dir": "models/cascade_v1",
            "stage1_keep": 31,
            "stage2_keep": 11,
        }
    }
    strategy_cfg = {
        "selected_strategy": {
            "version_id": "v0_binary_baseline",
            "stage_type": "baseline",
        }
    }
    metadata = {
        "selected_strategy": {
            "version_id": "v3_rerank_k30_p300",
            "stage_type": "rerank",
        }
    }
    strat, source = resolve_runtime_strategy(predict_cfg, strategy_cfg, metadata)
    assert source == "predict.yaml pipeline override"
    assert strat.stage_type == "cascade"
    assert strat.stage1_keep == 31
    assert strat.stage2_keep == 11


def test_predict_strategy_precedence_auto_fallback_to_train_pipeline() -> None:
    predict_cfg = {
        "pipeline": {
            "version": "auto",
            "artifact_dir": "models/cascade_v1",
            "stage1_keep": 30,
            "stage2_keep": 10,
        }
    }
    strategy_cfg = {}
    metadata = {}
    train_cfg = {"pipeline": {"version": "cascade_v1"}}
    strat, source = resolve_runtime_strategy(
        predict_cfg,
        strategy_cfg,
        metadata,
        train_cfg=train_cfg,
    )
    assert source == "train.yaml pipeline fallback"
    assert strat.stage_type == "cascade"


def test_train_config_default_pipeline_is_cascade_v1() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    assert str(cfg.get("pipeline", {}).get("version")) == "cascade_v1"

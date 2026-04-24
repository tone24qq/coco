from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest_weak_signal_ablation import check_lineage_or_board_leakage, run_ablation
from src.features.weak_signal import (
    build_weak_signal_features,
    compute_module_disagreement_signal,
    compute_neighbor_residual_signal,
    compute_residue_tail_signal,
)


def _mock_split() -> dict[str, pd.DataFrame]:
    rows = []
    for split, g0 in (("train", 0), ("valid", 100), ("holdout", 200)):
        for gid in range(g0, g0 + 3):
            for i in range(4):
                label = 1 if i == (gid % 4) else 0
                rows.append(
                    {
                        "group_id": f"g{gid}",
                        "lineage_id": f"lin_{gid}",
                        "board_id": f"b_{gid}",
                        "split": split,
                        "size_class": "8x10" if gid % 2 == 0 else "10x12",
                        "is_feasible": 1,
                        "cand_row": i,
                        "cand_col": 3 - i,
                        "label": label,
                        "baseline_score": 0.1 * i + (0.3 if label else 0.0),
                        "candidate_delta_residue_same_tail_count_local5x5": float(i),
                        "candidate_delta_neighbor_residue_same_tail_ratio": float(i) / 4.0,
                        "board_state_global_residue_entropy": 0.2 * i,
                        "module_consensus_top1": float(4 - i),
                        "module_consensus_top3": float(4 - i),
                        "module_consensus_top5": float(4 - i),
                        "mean_score": 1.0 - 0.1 * i,
                        "std_score": 0.1 * i,
                        "score_spread": 0.2 * i,
                        "disagreement_count": float(i),
                        "rank_entropy_like": 0.05 * i,
                        "conflict_mass": 0.03 * i,
                        "is_border": int(i in (0, 3)),
                        "is_corner": int(i in (0, 3)),
                        "row_norm": float(i) / 4.0,
                        "col_norm": float(3 - i) / 4.0,
                        "dist_to_center": abs(1.5 - i) / 3.0,
                    }
                )
    df = pd.DataFrame(rows)
    return {k: v.reset_index(drop=True) for k, v in df.groupby("split")}


def test_weak_signal_score_in_01_and_no_nan_inf() -> None:
    df = pd.DataFrame(
        {
            "group_id": ["g1", "g1", "g2", "g2"],
            "candidate_delta_residue_x": [0.0, 1.0, 2.0, 3.0],
            "candidate_delta_neighbor_x": [1, 2, 3, 4],
            "module_consensus_top1": [1, 0, 1, 0],
            "module_consensus_top3": [1, 0, 1, 0],
            "module_consensus_top5": [1, 0, 1, 0],
            "mean_score": [0.8, 0.2, 0.7, 0.3],
            "std_score": [0.1, 0.3, 0.1, 0.4],
            "score_spread": [0.2, 0.5, 0.2, 0.6],
            "disagreement_count": [0, 2, 0, 3],
            "rank_entropy_like": [0.1, 0.2, 0.1, 0.3],
            "conflict_mass": [0.1, 0.3, 0.1, 0.4],
            "is_border": [1, 0, 1, 0],
            "is_corner": [1, 0, 0, 1],
            "row_norm": [0.1, 0.8, 0.2, 0.9],
            "col_norm": [0.2, 0.7, 0.3, 0.6],
            "dist_to_center": [0.9, 0.1, 0.8, 0.2],
        }
    )
    out = build_weak_signal_features(df, {"normalization": {"global_minmax": True, "group_minmax": True}})
    assert out["weak_signal_score"].between(0.0, 1.0).all()
    assert np.isfinite(out[["weak_signal_score", "residue_tail_signal", "geometry_low_signal"]].to_numpy()).all()


def test_weak_signal_does_not_depend_on_label() -> None:
    base = pd.DataFrame({"group_id": ["g1", "g1"], "candidate_delta_residue_x": [0.0, 1.0], "label": [0, 1]})
    a = build_weak_signal_features(base, {})["weak_signal_score"].tolist()
    base["label"] = [1, 0]
    b = build_weak_signal_features(base, {})["weak_signal_score"].tolist()
    assert a == b


def test_optional_columns_missing_returns_zero_and_no_crash() -> None:
    df = pd.DataFrame({"group_id": ["g1", "g1"]})
    assert (compute_residue_tail_signal(df) == 0.0).all()
    assert (compute_neighbor_residual_signal(df) == 0.0).all()
    assert (compute_module_disagreement_signal(df) == 0.0).all()


def test_ablation_outputs_and_weight_zero_baseline(tmp_path: Path) -> None:
    splits = _mock_split()
    cfg = {
        "weights": [0.0, 0.05],
        "normalization": {"global_minmax": True, "group_minmax": True},
        "ensemble_weights": {
            "residue_tail_signal": 0.3,
            "geometry_low_signal": 0.25,
            "neighbor_residual_signal": 0.25,
            "module_disagreement_signal": 0.2,
        },
    }
    decision = run_ablation(splits, cfg, tmp_path)

    summary = pd.read_csv(tmp_path / "summary.csv")
    per_group = pd.read_csv(tmp_path / "per_group.csv")

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "per_group.csv").exists()
    assert (tmp_path / "per_size.csv").exists()
    assert (tmp_path / "debug_input.json").exists()
    assert (tmp_path / "decision.json").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "README.md").exists()

    baseline = summary.loc[summary["weight"] == 0.0].iloc[0]
    from_per_group = per_group[per_group["weight"] == 0.0]
    assert abs(float(baseline["top1"]) - float((from_per_group["rank"] == 1).mean())) < 1e-12

    payload = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert "accepted" in payload
    assert "best_weight" in payload
    assert "guardrails" in payload
    assert payload["best_weight"] == decision["best_weight"]


def test_group_integrity_one_positive_per_group() -> None:
    splits = _mock_split()
    for frame in splits.values():
        assert (frame.groupby("group_id")["label"].sum() == 1).all()


def test_by_lineage_leakage_guard() -> None:
    train = pd.DataFrame({"lineage_id": ["A", "B"], "board_id": ["x", "y"]})
    holdout = pd.DataFrame({"lineage_id": ["C", "B"], "board_id": ["z", "w"]})
    assert not check_lineage_or_board_leakage(train, holdout)

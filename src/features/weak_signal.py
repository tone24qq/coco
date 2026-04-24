from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


EPS = 1e-12


def normalize_01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    vmin = float(values.min()) if len(values) else 0.0
    vmax = float(values.max()) if len(values) else 0.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) <= EPS:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return ((values - vmin) / (vmax - vmin)).clip(0.0, 1.0)


def _mean_available(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    use = [c for c in cols if c in df.columns]
    if not use:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return df[use].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).mean(axis=1)


def compute_residue_tail_signal(df: pd.DataFrame) -> pd.Series:
    cols: List[str] = [
        c
        for c in df.columns
        if (
            c.startswith("candidate_delta_") or c.startswith("board_state_")
        )
        and any(k in c for k in ("residue", "multiple10", "hist", "mode_bin", "tail"))
    ]
    if not cols:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return normalize_01(_mean_available(df, cols))


def _safe_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def compute_geometry_low_signal(df: pd.DataFrame) -> pd.Series:
    is_border = _safe_col(df, "is_border")
    is_corner = _safe_col(df, "is_corner")

    row_norm = _safe_col(df, "row_norm", np.nan)
    col_norm = _safe_col(df, "col_norm", np.nan)
    dist_to_center = _safe_col(df, "dist_to_center", np.nan)

    if row_norm.isna().any() and {"cand_row", "rows"}.issubset(df.columns):
        row_norm = _safe_col(df, "cand_row") / _safe_col(df, "rows").replace(0.0, 1.0)
    if col_norm.isna().any() and {"cand_col", "cols"}.issubset(df.columns):
        col_norm = _safe_col(df, "cand_col") / _safe_col(df, "cols").replace(0.0, 1.0)

    if dist_to_center.isna().any() and {"cand_row", "cand_col", "rows", "cols"}.issubset(df.columns):
        cr = _safe_col(df, "cand_row")
        cc = _safe_col(df, "cand_col")
        rows = _safe_col(df, "rows").replace(0.0, 1.0)
        cols = _safe_col(df, "cols").replace(0.0, 1.0)
        center_r = (rows + 1.0) / 2.0
        center_c = (cols + 1.0) / 2.0
        denom = (center_r + center_c).replace(0.0, 1.0)
        dist_to_center = ((cr - center_r).abs() + (cc - center_c).abs()) / denom

    row_norm = row_norm.fillna(0.0)
    col_norm = col_norm.fillna(0.0)
    dist_to_center = dist_to_center.fillna(0.0)

    center_bias = 1.0 - normalize_01(dist_to_center)
    edge_bias = normalize_01((is_border + is_corner) / 2.0)
    axis_balance = 1.0 - normalize_01((row_norm - 0.5).abs() + (col_norm - 0.5).abs())
    return normalize_01(0.45 * center_bias + 0.30 * edge_bias + 0.25 * axis_balance)


def compute_neighbor_residual_signal(df: pd.DataFrame) -> pd.Series:
    cols = [
        c
        for c in df.columns
        if any(k in c for k in ("local5x5_", "neighbor_", "row_", "col_"))
        and (
            c.startswith("candidate_delta_")
            or c.startswith("board_state_")
            or c.startswith("row_")
            or c.startswith("col_")
        )
    ]
    if not cols:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return normalize_01(_mean_available(df, cols))


def compute_module_disagreement_signal(df: pd.DataFrame) -> pd.Series:
    cols = [
        "module_consensus_top1",
        "module_consensus_top3",
        "module_consensus_top5",
        "mean_score",
        "std_score",
        "score_spread",
        "disagreement_count",
        "rank_entropy_like",
        "conflict_mass",
    ]
    data = {c: _safe_col(df, c, 0.0) for c in cols}
    consensus = (
        normalize_01(data["module_consensus_top1"])
        + normalize_01(data["module_consensus_top3"])
        + normalize_01(data["module_consensus_top5"])
    ) / 3.0
    disagreement = (
        normalize_01(data["std_score"])
        + normalize_01(data["score_spread"])
        + normalize_01(data["disagreement_count"])
        + normalize_01(data["rank_entropy_like"])
        + normalize_01(data["conflict_mass"])
    ) / 5.0
    mean_score = normalize_01(data["mean_score"])
    return normalize_01(0.50 * (1.0 - disagreement) + 0.30 * consensus + 0.20 * mean_score)


def _group_normalize(df: pd.DataFrame, col: str, group_col: str = "group_id") -> pd.Series:
    if group_col not in df.columns:
        return normalize_01(df[col])

    def _norm(g: pd.Series) -> pd.Series:
        return normalize_01(g)

    return df.groupby(group_col, sort=False)[col].transform(_norm)


def build_weak_signal_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    ensemble = config.get("ensemble_weights", {}) if isinstance(config, dict) else {}
    w_res = float(ensemble.get("residue_tail_signal", 0.30))
    w_geo = float(ensemble.get("geometry_low_signal", 0.25))
    w_nbr = float(ensemble.get("neighbor_residual_signal", 0.25))
    w_mod = float(ensemble.get("module_disagreement_signal", 0.20))

    out["residue_tail_signal"] = compute_residue_tail_signal(out)
    out["geometry_low_signal"] = compute_geometry_low_signal(out)
    out["neighbor_residual_signal"] = compute_neighbor_residual_signal(out)
    out["module_disagreement_signal"] = compute_module_disagreement_signal(out)

    out["weak_signal_score_raw"] = (
        w_res * out["residue_tail_signal"]
        + w_geo * out["geometry_low_signal"]
        + w_nbr * out["neighbor_residual_signal"]
        + w_mod * out["module_disagreement_signal"]
    )

    norm_cfg = config.get("normalization", {}) if isinstance(config, dict) else {}
    if bool(norm_cfg.get("global_minmax", True)):
        out["weak_signal_score"] = normalize_01(out["weak_signal_score_raw"])
    else:
        out["weak_signal_score"] = out["weak_signal_score_raw"].copy()

    if bool(norm_cfg.get("group_minmax", True)):
        out["weak_signal_score_group"] = _group_normalize(out, "weak_signal_score")
        out["weak_signal_score"] = normalize_01(0.5 * out["weak_signal_score"] + 0.5 * out["weak_signal_score_group"])

    signal_cols = [
        "residue_tail_signal",
        "geometry_low_signal",
        "neighbor_residual_signal",
        "module_disagreement_signal",
        "weak_signal_score_raw",
        "weak_signal_score",
    ]
    for col in signal_cols:
        s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[col] = s.clip(0.0, 1.0) if col != "weak_signal_score_raw" else normalize_01(s)

    return out

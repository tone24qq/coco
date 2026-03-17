from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.features import (
    candidate_analysis_compatibility_score,
    derive_analysis_target_profile,
)
from src.utils import apply_local_peak_correction, apply_topk_group_dedup


@dataclass
class RuntimeScoringOutputs:
    score_table: pd.DataFrame
    rerank_summary: dict
    local_peak_summary: dict
    dedup_summary: dict


def _normalize_series(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    denom = float(arr.std(ddof=0))
    if denom <= 1e-9:
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return (arr - float(arr.mean())) / denom


def normalize_rank_pct(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    if len(arr) == 0:
        return arr
    return arr.rank(method="average", pct=True).astype(float)


def history_prior_from_snapshot(
    score_table: pd.DataFrame,
    snapshot_payload: dict,
) -> pd.DataFrame:
    out = score_table.copy()
    priors = snapshot_payload.get("number_priors")
    if priors is None or priors.empty:
        out["history_prior_score"] = 0.0
        return out

    cols = [
        "total_hits_all_time",
        "hits_last_200",
        "hits_last_500",
        "hits_last_1000",
        "today_hits",
        "carryover_from_prev",
        "pm1_neighbor_hits",
        "pm2_neighbor_hits",
        "current_gap_all",
        "avg_gap_all",
        "max_gap_all",
    ]
    priors_df = priors.reset_index(drop=True)
    merged = out.merge(priors_df[["number", *cols]], on="number", how="left")
    merged = merged.fillna(0.0)

    positive = (
        0.28 * _normalize_series(merged["total_hits_all_time"])
        + 0.18 * _normalize_series(merged["hits_last_200"])
        + 0.12 * _normalize_series(merged["hits_last_500"])
        + 0.10 * _normalize_series(merged["hits_last_1000"])
        + 0.07 * _normalize_series(merged["today_hits"])
        + 0.08 * _normalize_series(merged["carryover_from_prev"])
        + 0.09 * _normalize_series(merged["pm1_neighbor_hits"])
        + 0.08 * _normalize_series(merged["pm2_neighbor_hits"])
    )
    penalty = (
        0.30 * _normalize_series(merged["current_gap_all"])
        + 0.10 * _normalize_series(merged["avg_gap_all"])
        + 0.05 * _normalize_series(merged["max_gap_all"])
    )
    merged["history_prior_score"] = (positive - penalty).astype(float)
    return merged


def analysis_rerank(
    score_table: pd.DataFrame,
    recent_draws: list[list[int]],
    board_priors: dict,
    top_k: int,
    rerank_weight: float,
    enabled: bool,
) -> tuple[pd.DataFrame, dict]:
    work = score_table.copy()
    work["analysis_compatibility_score"] = 0.0
    work["analysis_rerank_score"] = 0.0
    work["score_before_analysis_rerank"] = work["final_score"].astype(float)

    summary = {
        "enabled": False,
        "top_k": int(top_k),
        "weight": float(rerank_weight),
        "target_profile": {},
        "top_k_preview": [],
    }
    if not enabled:
        work["score_after_analysis_rerank"] = work["final_score"].astype(float)
        return work, summary

    profile = derive_analysis_target_profile(recent_draws, board_priors=board_priors)
    top_k = max(3, min(int(top_k), len(work)))
    top = work.head(top_k).copy()
    top["analysis_compatibility_score"] = top["number"].apply(
        lambda n: candidate_analysis_compatibility_score(int(n), profile)
    )
    top["analysis_rerank_score"] = float(rerank_weight) * top[
        "analysis_compatibility_score"
    ].astype(float)
    top["final_score"] = (
        top["score_before_analysis_rerank"] + top["analysis_rerank_score"]
    )

    tail = work.iloc[top_k:].copy()
    reranked = pd.concat([top, tail], ignore_index=True)
    reranked["score_after_analysis_rerank"] = reranked["final_score"].astype(float)
    reranked = reranked.sort_values("final_score", ascending=False).reset_index(
        drop=True
    )

    summary = {
        "enabled": True,
        "top_k": int(top_k),
        "weight": float(rerank_weight),
        "target_profile": profile,
        "top_k_preview": reranked.head(5)[
            [
                "number",
                "score_before_analysis_rerank",
                "analysis_compatibility_score",
                "analysis_rerank_score",
                "score_after_analysis_rerank",
            ]
        ].to_dict(orient="records"),
    }
    return reranked, summary


def score_candidates_runtime(
    *,
    base_scores: np.ndarray,
    candidate_df: pd.DataFrame,
    recent_draws: list[list[int]],
    runtime_config: dict,
    snapshot_payload: dict,
    board_priors: dict,
    soft_label_raw: np.ndarray | None,
    pm1_proximity_raw: np.ndarray | None,
    local_peak_cfg_override: dict | None = None,
    dedup_cfg_override: dict | None = None,
) -> RuntimeScoringOutputs:
    score_table = pd.DataFrame(
        {"number": list(range(1, 81)), "model_score": [float(x) for x in base_scores]}
    )
    score_table = score_table.sort_values("model_score", ascending=False).reset_index(
        drop=True
    )

    history_cfg = runtime_config.get("history_prior", {})
    history_enabled = bool(history_cfg.get("enabled", True))
    model_weight = float(history_cfg.get("model_weight", 0.88))
    history_weight = float(history_cfg.get("history_weight", 0.12))

    long_cfg = runtime_config.get("long_feature_injection", {})
    long_enabled = bool(long_cfg.get("enabled", True))
    long_weight = float(long_cfg.get("weight", 0.06))

    score_table = history_prior_from_snapshot(score_table, snapshot_payload)
    if not history_enabled:
        score_table["history_prior_score"] = 0.0

    long_cols = [
        "cand_hits_last_200",
        "cand_hits_last_500",
        "cand_hits_last_1000",
        "cand_total_hits_all_time",
        "cand_current_gap_all",
        "cand_avg_gap_all",
        "cand_max_gap_all",
        "cand_today_hits",
        "cand_carryover_from_prev",
        "cand_pm1_neighbor_hits",
        "cand_pm2_neighbor_hits",
    ]
    score_table = score_table.merge(
        candidate_df[["number", *long_cols]], on="number", how="left"
    ).fillna(0.0)

    long_positive = (
        0.30 * _normalize_series(score_table["cand_hits_last_200"])
        + 0.20 * _normalize_series(score_table["cand_hits_last_500"])
        + 0.15 * _normalize_series(score_table["cand_hits_last_1000"])
        + 0.10 * _normalize_series(score_table["cand_total_hits_all_time"])
        + 0.10 * _normalize_series(score_table["cand_today_hits"])
        + 0.08 * _normalize_series(score_table["cand_carryover_from_prev"])
        + 0.04 * _normalize_series(score_table["cand_pm1_neighbor_hits"])
        + 0.03 * _normalize_series(score_table["cand_pm2_neighbor_hits"])
    )
    long_penalty = (
        0.45 * _normalize_series(score_table["cand_current_gap_all"])
        + 0.20 * _normalize_series(score_table["cand_avg_gap_all"])
        + 0.10 * _normalize_series(score_table["cand_max_gap_all"])
    )
    score_table["long_feature_score"] = (long_positive - long_penalty).astype(float)
    if not long_enabled:
        score_table["long_feature_score"] = 0.0

    soft_cfg = runtime_config.get("soft_label_training", {})
    soft_enabled = bool(soft_cfg.get("enabled", False)) and soft_label_raw is not None
    soft_weight = float(soft_cfg.get("blend_weight", 0.15)) if soft_enabled else 0.0
    soft_norm = str(soft_cfg.get("normalization", "rank_pct"))
    if soft_enabled:
        soft_series = pd.Series([float(x) for x in soft_label_raw])
        if soft_norm == "rank_pct":
            score_table["soft_label_score"] = normalize_rank_pct(soft_series)
        else:
            score_table["soft_label_score"] = _normalize_series(soft_series)
    else:
        score_table["soft_label_score"] = 0.0

    pm1_cfg = runtime_config.get("proximity_model", {})
    pm1_enabled = bool(pm1_cfg.get("enabled", False)) and pm1_proximity_raw is not None
    pm1_weight = float(pm1_cfg.get("pm1_weight", 0.12)) if pm1_enabled else 0.0
    score_table["pm1_proximity_score"] = (
        pd.Series([float(x) for x in pm1_proximity_raw]) if pm1_enabled else 0.0
    )

    score_table["exact_model_score"] = score_table["model_score"].astype(float)
    score_table["final_score"] = (
        model_weight * score_table["model_score"]
        + history_weight * score_table["history_prior_score"]
        + long_weight * score_table["long_feature_score"]
        + soft_weight * score_table["soft_label_score"]
        + pm1_weight * score_table["pm1_proximity_score"]
    )

    rerank_cfg = runtime_config.get("analysis_rerank", {})
    score_table, rerank_summary = analysis_rerank(
        score_table,
        recent_draws=recent_draws,
        board_priors=board_priors,
        top_k=int(rerank_cfg.get("top_k", 30)),
        rerank_weight=float(rerank_cfg.get("weight", 0.08)),
        enabled=bool(rerank_cfg.get("enabled", True)),
    )

    score_table, local_peak_summary = apply_local_peak_correction(
        score_table,
        cfg=(
            local_peak_cfg_override
            if local_peak_cfg_override is not None
            else runtime_config.get("neighbor_peak_correction", {})
        ),
        input_score_column="score_after_analysis_rerank",
        output_score_column="score_after_local_peak",
    )
    score_table["final_score"] = score_table["score_after_local_peak"].astype(float)

    score_table, dedup_summary = apply_topk_group_dedup(
        score_table,
        cfg=(
            dedup_cfg_override
            if dedup_cfg_override is not None
            else runtime_config.get("topk_group_dedup", {})
        ),
        top_k=3,
    )

    score_table = score_table.sort_values("final_score", ascending=False).reset_index(
        drop=True
    )
    score_table["rank_final"] = np.arange(1, len(score_table) + 1, dtype=int)
    score_table["rank_model_only"] = (
        score_table["model_score"].rank(method="min", ascending=False).astype(int)
    )
    score_table["score"] = score_table["final_score"].astype(float)

    return RuntimeScoringOutputs(
        score_table=score_table,
        rerank_summary=rerank_summary,
        local_peak_summary=local_peak_summary,
        dedup_summary=dedup_summary,
    )

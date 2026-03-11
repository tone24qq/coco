from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    version_id: str
    stage_type: str
    candidate_pool: int
    prior_window: int
    rerank_weight: float
    penalty_weight: float
    trend_weight: float
    regime_aware: bool = True


def default_experiments() -> list[StrategyConfig]:
    return [
        StrategyConfig("v0_binary_baseline", "baseline", 20, 100, 0.0, 0.0, 0.0, False),
        StrategyConfig("v1_rerank_k20_p100", "rerank", 20, 100, 2.2, 0.08, 0.25, True),
        StrategyConfig("v2_rerank_k30_p300", "rerank", 30, 300, 3.2, 0.10, 0.35, True),
        StrategyConfig("v3_rerank_k40_p500", "rerank", 40, 500, 4.2, 0.12, 0.45, True),
        StrategyConfig(
            "v4_two_stage_20_10_3", "two_stage", 20, 300, 3.0, 0.11, 0.4, True
        ),
    ]


def derive_regime(row: pd.Series) -> str:
    if float(row.get("span", 0)) >= 72 or float(row.get("consecutive_pairs", 0)) >= 6:
        return "high_vol"
    if float(row.get("zone_range", 0)) <= 2 and float(row.get("span", 0)) <= 58:
        return "balanced"
    return "transitional"


def _prior_feature_name(prior_window: int) -> str:
    return f"freq_last_{prior_window}"


def _safe_col(cand: pd.DataFrame, name: str) -> np.ndarray:
    if name in cand.columns:
        return cand[name].to_numpy(dtype=float)
    return np.zeros(len(cand), dtype=float)


def apply_strategy(
    base_scores: np.ndarray,
    cand: pd.DataFrame,
    cfg: StrategyConfig,
    regime: str,
) -> np.ndarray:
    if cfg.stage_type == "baseline":
        return base_scores

    scores = base_scores.copy()
    if cfg.stage_type == "rerank":
        return _rerank_once(scores, cand, cfg.candidate_pool, cfg, regime)

    scores = _rerank_once(scores, cand, cfg.candidate_pool, cfg, regime)
    scores = _rerank_once(scores, cand, 10, cfg, regime, scale=0.9)
    scores = _rerank_once(scores, cand, 3, cfg, regime, scale=1.1)
    return scores


def _rerank_once(
    scores: np.ndarray,
    cand: pd.DataFrame,
    pool_k: int,
    cfg: StrategyConfig,
    regime: str,
    scale: float = 1.0,
) -> np.ndarray:
    out = scores.copy()
    idx = np.argsort(out)[::-1][:pool_k]
    c = cand.iloc[idx]

    prior_col = _prior_feature_name(cfg.prior_window)
    prior_signal = _safe_col(c, prior_col)
    trend = _safe_col(c, "ema_short_minus_ema_long")
    cooccur = _safe_col(c, "cooccur_with_last_draw_mean")
    zone = _safe_col(c, "num_zone")

    regime_boost = 1.0
    if cfg.regime_aware:
        regime_boost = {"high_vol": 1.15, "balanced": 1.0, "transitional": 1.05}.get(
            regime, 1.0
        )

    bonus = 0.6 * prior_signal + cfg.trend_weight * trend + 0.2 * cooccur
    penalty = cfg.penalty_weight * np.abs(zone - np.median(zone))
    out[idx] = (
        out[idx] + scale * cfg.rerank_weight * 0.01 * regime_boost * bonus - penalty
    )
    return out


def top_hits(scores: np.ndarray, actual: set[int]) -> tuple[int, int, int]:
    order = np.argsort(scores)[::-1] + 1
    return (
        len(set(order[:20]) & actual),
        len(set(order[:10]) & actual),
        len(set(order[:3]) & actual),
    )


def issue_metrics(scores: np.ndarray, actual: set[int]) -> dict[str, float]:
    h20, h10, h3 = top_hits(scores, actual)
    return {
        "top20_hit_rate": h20 / 20,
        "top10_hit_rate": h10 / 10,
        "top3_hit_rate": h3 / 3,
        "top3_at_least_one_hit_rate": float(h3 > 0),
    }


def strategy_to_dict(cfg: StrategyConfig) -> dict[str, Any]:
    return {
        "version_id": cfg.version_id,
        "stage_type": cfg.stage_type,
        "candidate_pool": cfg.candidate_pool,
        "prior_window": cfg.prior_window,
        "rerank_weight": cfg.rerank_weight,
        "penalty_weight": cfg.penalty_weight,
        "trend_weight": cfg.trend_weight,
        "regime_aware": cfg.regime_aware,
    }

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class RuntimeWeights:
    ranker: float
    logistic: float
    retrieval: float
    history_prior: float
    analysis: float
    local_peak: float

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> "RuntimeWeights":
        return cls(
            ranker=float(values.get("ranker", 0.55)),
            logistic=float(values.get("logistic", 0.2)),
            retrieval=float(values.get("retrieval", 0.15)),
            history_prior=float(values.get("history_prior", 0.1)),
            analysis=float(values.get("analysis", 0.0)),
            local_peak=float(values.get("local_peak", 0.0)),
        )


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def analysis_rerank_component(rows: pd.DataFrame) -> pd.Series:
    gap_bonus = rows["cand_current_gap"].clip(lower=1).rdiv(1.0)
    neighbor_bonus = (rows["cand_pm1_neighbor_hits"] + rows["cand_pm2_neighbor_hits"]).clip(lower=0)
    return _minmax(0.7 * gap_bonus + 0.3 * neighbor_bonus)


def local_peak_component(rows: pd.DataFrame) -> pd.Series:
    rebound = rows["cand_rebound_score"].clip(lower=0)
    reactivate = rows["cand_recent_reactivation_score"].clip(lower=0)
    return _minmax(0.5 * rebound + 0.5 * reactivate)


def score_candidates(
    rows: pd.DataFrame,
    ranker_score: Iterable[float],
    logistic_score: Iterable[float],
    weights: RuntimeWeights,
) -> pd.DataFrame:
    table = rows.copy()
    table["ranker_score"] = list(ranker_score)
    table["logistic_score"] = list(logistic_score)
    table["retrieval_score"] = table["retrieval_next_draw_posterior"].astype(float)
    raw_prior = table["cand_hits_last_100"].astype(float)
    table["history_prior_score"] = _minmax(raw_prior)
    table["analysis_rerank_score"] = analysis_rerank_component(table)
    table["local_peak_score"] = local_peak_component(table)

    table["final_score"] = (
        weights.ranker * table["ranker_score"]
        + weights.logistic * table["logistic_score"]
        + weights.retrieval * table["retrieval_score"]
        + weights.history_prior * table["history_prior_score"]
        + weights.analysis * table["analysis_rerank_score"]
        + weights.local_peak * table["local_peak_score"]
    )

    table = table.sort_values(["issue", "final_score"], ascending=[True, False]).reset_index(drop=True)
    table["rank_final"] = table.groupby("issue").cumcount() + 1
    return table

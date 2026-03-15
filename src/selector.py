from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectorContext:
    regime: str
    allow_single_zone_focus: bool
    prefer_balance: bool
    prefer_two_zone_mix: bool


def build_selector_context(issue_row: pd.Series) -> SelectorContext:
    entropy = float(issue_row.get("issue_zone_entropy", 0.0))
    span_z = float(issue_row.get("issue_span_z50", 0.0))
    consecutive = float(issue_row.get("issue_consecutive_z50", 0.0))
    if entropy >= 0.9 and span_z <= 0.1:
        regime = "balanced"
    elif span_z >= 1.2 or consecutive >= 1.0:
        regime = "single_zone_burst"
    else:
        regime = "double_zone_shake"
    return SelectorContext(
        regime=regime,
        allow_single_zone_focus=regime == "single_zone_burst",
        prefer_balance=regime == "balanced",
        prefer_two_zone_mix=regime == "double_zone_shake",
    )


@dataclass(frozen=True)
class SelectorWeights:
    stage2_weight: float = 1.0
    stage1_aux_weight: float = 0.2
    diversity_weight: float = 0.25
    zone_weight: float = 0.20
    tail_weight: float = 0.15


@dataclass
class CombinationSelectionResult:
    final_top3: list[int]
    selector_score: float
    reason: str
    scored_table: pd.DataFrame


def _combo_metrics(combo_df: pd.DataFrame) -> dict[str, float]:
    nums = combo_df["number"].astype(int).tolist()
    dists = [abs(a - b) for a, b in combinations(nums, 2)]
    min_dist = float(min(dists)) if dists else 0.0
    mean_dist = float(np.mean(dists)) if dists else 0.0
    zones = combo_df["zone_id"].astype(int).tolist()
    tails = combo_df["tail"].astype(int).tolist()
    zone_unique = len(set(zones))
    tail_unique = len(set(tails))
    same_zone = float(zone_unique == 1)
    same_tail = float(tail_unique == 1)
    return {
        "min_pair_distance": min_dist,
        "mean_pair_distance": mean_dist,
        "zone_unique": float(zone_unique),
        "tail_unique": float(tail_unique),
        "is_same_zone": same_zone,
        "is_same_tail": same_tail,
    }


def score_top3_combinations(
    stage3_inputs: pd.DataFrame,
    context: SelectorContext,
    weights: SelectorWeights | None = None,
) -> pd.DataFrame:
    if len(stage3_inputs) < 3:
        raise ValueError("stage3_inputs must contain at least 3 candidates")
    w = weights or SelectorWeights()
    rows = []
    cols = [
        c
        for c in ["number", "zone_id", "tail", "stage2_score", "stage1_score"]
        if c in stage3_inputs.columns
    ]
    for idx_tuple in combinations(range(len(stage3_inputs)), 3):
        combo_df = stage3_inputs.iloc[list(idx_tuple)][cols].copy()
        m = _combo_metrics(combo_df)
        s2 = float(combo_df["stage2_score"].mean())
        s1 = (
            float(combo_df["stage1_score"].mean())
            if "stage1_score" in combo_df.columns
            else 0.0
        )

        diversity = min(1.0, m["mean_pair_distance"] / 20.0)
        zone_term = 0.0
        if context.prefer_balance:
            zone_term = (m["zone_unique"] - 1.0) / 2.0
        elif context.allow_single_zone_focus:
            zone_term = 1.0 - m["is_same_zone"] * 0.25
        elif context.prefer_two_zone_mix:
            zone_term = 1.0 if m["zone_unique"] == 2.0 else 0.7

        tail_penalty = m["is_same_tail"] * 0.7
        if context.allow_single_zone_focus:
            tail_penalty *= 0.6

        score = (
            w.stage2_weight * s2
            + w.stage1_aux_weight * s1
            + w.diversity_weight * diversity
            + w.zone_weight * zone_term
            + w.tail_weight * (m["tail_unique"] / 3.0)
            - tail_penalty
        )
        nums = sorted(int(x) for x in combo_df["number"].tolist())
        rows.append(
            {
                "numbers": nums,
                "selector_score": float(score),
                "stage2_mean": s2,
                "stage1_mean": s1,
                **m,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("selector_score", ascending=False)
        .reset_index(drop=True)
    )


def select_top3_combination(
    stage3_inputs: pd.DataFrame,
    context: SelectorContext,
) -> CombinationSelectionResult:
    ranked = score_top3_combinations(stage3_inputs, context)
    best = ranked.iloc[0]
    numbers = [int(x) for x in best["numbers"]]
    reason = (
        f"regime={context.regime}; stage2_mean={best['stage2_mean']:.4f}; "
        f"zone_unique={int(best['zone_unique'])}; tail_unique={int(best['tail_unique'])}; "
        f"mean_dist={best['mean_pair_distance']:.2f}"
    )
    return CombinationSelectionResult(
        final_top3=numbers,
        selector_score=float(best["selector_score"]),
        reason=reason,
        scored_table=ranked,
    )

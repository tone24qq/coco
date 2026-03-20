from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

import pandas as pd

from src.utils import DataContractError


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


@dataclass(frozen=True)
class DynamicWeightConfig:
    enabled: bool
    mode: str
    ranker_min: float
    logistic_min: float
    retrieval_min: float
    history_prior_fixed: float
    analysis_fixed: float
    local_peak_fixed: float

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "DynamicWeightConfig":
        if values is None:
            return cls(
                enabled=False,
                mode="disabled",
                ranker_min=0.35,
                logistic_min=0.08,
                retrieval_min=0.10,
                history_prior_fixed=0.10,
                analysis_fixed=0.03,
                local_peak_fixed=0.02,
            )
        try:
            min_w = values["min_weights"]
            fixed_w = values["fixed_weights"]
            cfg = cls(
                enabled=bool(values["enabled"]),
                mode=str(values["mode"]),
                ranker_min=float(min_w["ranker"]),
                logistic_min=float(min_w["logistic"]),
                retrieval_min=float(min_w["retrieval"]),
                history_prior_fixed=float(fixed_w["history_prior"]),
                analysis_fixed=float(fixed_w["analysis"]),
                local_peak_fixed=float(fixed_w["local_peak"]),
            )
        except Exception as exc:  # noqa: BLE001
            raise DataContractError(f"runtime_scoring.dynamic config invalid: {exc}") from exc
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.enabled and self.mode != "heuristic_retrieval_gate_v1":
            raise DataContractError(f"unsupported dynamic weighting mode: {self.mode}")
        if min(self.ranker_min, self.logistic_min, self.retrieval_min) < 0:
            raise DataContractError("dynamic min_weights must be >= 0")
        if self.history_prior_fixed + self.analysis_fixed + self.local_peak_fixed >= 1.0:
            raise DataContractError("dynamic fixed_weights sum must be < 1.0")


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


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _require_retrieval_columns(rows: pd.DataFrame) -> None:
    needed = [
        "retrieval_similarity_max",
        "retrieval_similarity_mean",
        "retrieval_next_draw_posterior",
        "retrieval_exact_draw_match_count_mean",
        "retrieval_dynamic_context_n",
        "retrieval_same_day_progress_bonus",
    ]
    missing = [c for c in needed if c not in rows.columns]
    if missing:
        raise DataContractError(f"missing retrieval columns for dynamic weighting: {missing}")


def resolve_issue_dynamic_weights(
    issue_rows: pd.DataFrame,
    base_weights: RuntimeWeights,
    dynamic_cfg: DynamicWeightConfig,
) -> tuple[dict[str, float], float]:
    if not dynamic_cfg.enabled:
        eff = {
            "ranker": base_weights.ranker,
            "logistic": base_weights.logistic,
            "retrieval": base_weights.retrieval,
            "history_prior": base_weights.history_prior,
            "analysis": base_weights.analysis,
            "local_peak": base_weights.local_peak,
        }
        total = sum(eff.values())
        if total <= 0:
            raise DataContractError(f"fixed runtime weights must have positive sum, got {total}")
        return eff, 0.0

    _require_retrieval_columns(issue_rows)
    rows = issue_rows
    sim_max = float(rows["retrieval_similarity_max"].mean(skipna=False))
    sim_mean = float(rows["retrieval_similarity_mean"].mean(skipna=False))
    posterior = float(rows["retrieval_next_draw_posterior"].mean(skipna=False))
    exact = float(rows["retrieval_exact_draw_match_count_mean"].mean(skipna=False))
    context_n = float(rows["retrieval_dynamic_context_n"].mean(skipna=False))
    same_day = float(rows["retrieval_same_day_progress_bonus"].mean(skipna=False))
    if any(not isfinite(v) for v in [sim_max, sim_mean, posterior, exact, context_n, same_day]):
        raise DataContractError("dynamic weighting inputs contain NaN/inf")
    if context_n <= 0:
        raise DataContractError("retrieval_dynamic_context_n must be > 0 for dynamic weighting")
    exact_ratio = _clip(exact / max(context_n, 1.0), 0.0, 1.0)
    g_raw = 0.30 * sim_max + 0.20 * sim_mean + 0.20 * posterior + 0.15 * exact_ratio + 0.15 * same_day
    g = _clip(g_raw, 0.0, 1.0)
    if not isfinite(g):
        raise DataContractError("dynamic gate g is NaN/inf")

    raw_ranker = 0.58 - 0.18 * g
    raw_logistic = 0.17 - 0.04 * g
    raw_retrieval = 0.10 + 0.25 * g

    clipped_ranker = max(dynamic_cfg.ranker_min, raw_ranker)
    clipped_logistic = max(dynamic_cfg.logistic_min, raw_logistic)
    clipped_retrieval = max(dynamic_cfg.retrieval_min, raw_retrieval)
    active_total = clipped_ranker + clipped_logistic + clipped_retrieval
    if active_total <= 0:
        raise DataContractError("dynamic active weights invalid: non-positive sum")

    dynamic_total = 1.0 - (
        dynamic_cfg.history_prior_fixed + dynamic_cfg.analysis_fixed + dynamic_cfg.local_peak_fixed
    )
    ranker = clipped_ranker / active_total * dynamic_total
    logistic = clipped_logistic / active_total * dynamic_total
    retrieval = clipped_retrieval / active_total * dynamic_total
    eff = {
        "ranker": ranker,
        "logistic": logistic,
        "retrieval": retrieval,
        "history_prior": dynamic_cfg.history_prior_fixed,
        "analysis": dynamic_cfg.analysis_fixed,
        "local_peak": dynamic_cfg.local_peak_fixed,
    }
    total = sum(eff.values())
    if abs(total - 1.0) > 1e-9:
        raise DataContractError(f"dynamic runtime weights sum invalid: {total}")
    return eff, g


def compose_final_score_from_components(
    table: pd.DataFrame,
    base_weights: RuntimeWeights,
    dynamic_cfg: DynamicWeightConfig,
    return_diagnostics: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]] | pd.DataFrame:
    needed = [
        "issue",
        "ranker_score",
        "logistic_score",
        "retrieval_score",
        "history_prior_score",
        "analysis_rerank_score",
        "local_peak_score",
    ]
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise DataContractError(f"component columns missing for final compose: {missing}")
    out = table.copy()
    diagnostics: dict[str, Any] = {"dynamic_weighting_enabled": dynamic_cfg.enabled, "issues": {}}
    finals = []
    for issue, grp in out.groupby("issue", sort=False):
        eff, gate = resolve_issue_dynamic_weights(grp, base_weights, dynamic_cfg)
        if any(not isfinite(v) for v in eff.values()):
            raise DataContractError("effective runtime weights contain NaN/inf")
        part = grp.copy()
        part["final_score"] = (
            eff["ranker"] * part["ranker_score"]
            + eff["logistic"] * part["logistic_score"]
            + eff["retrieval"] * part["retrieval_score"]
            + eff["history_prior"] * part["history_prior_score"]
            + eff["analysis"] * part["analysis_rerank_score"]
            + eff["local_peak"] * part["local_peak_score"]
        )
        finals.append(part)
        diagnostics["issues"][str(issue)] = {"gate_value": gate, "effective_weights": eff}
    out = pd.concat(finals, ignore_index=True)
    out = out.sort_values(["issue", "final_score"], ascending=[True, False]).reset_index(drop=True)
    out["rank_final"] = out.groupby("issue").cumcount() + 1
    if return_diagnostics:
        return out, diagnostics
    return out


def score_candidates(
    rows: pd.DataFrame,
    ranker_score: Iterable[float],
    logistic_score: Iterable[float],
    weights: RuntimeWeights,
    dynamic_cfg: DynamicWeightConfig | None = None,
    return_diagnostics: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]] | pd.DataFrame:
    table = rows.copy()
    table["ranker_score"] = list(ranker_score)
    table["logistic_score"] = list(logistic_score)
    table["retrieval_score"] = table["retrieval_next_draw_posterior"].astype(float)
    raw_prior = table["cand_hits_last_100"].astype(float)
    table["history_prior_score"] = _minmax(raw_prior)
    table["analysis_rerank_score"] = analysis_rerank_component(table)
    table["local_peak_score"] = local_peak_component(table)

    cfg = dynamic_cfg or DynamicWeightConfig.from_mapping(None)
    return compose_final_score_from_components(
        table=table,
        base_weights=weights,
        dynamic_cfg=cfg,
        return_diagnostics=return_diagnostics,
    )

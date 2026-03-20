import numpy as np
import pandas as pd
import pytest

from src.runtime_scoring import (
    DynamicWeightConfig,
    RuntimeWeights,
    compose_final_score_from_components,
    resolve_issue_dynamic_weights,
    score_candidates,
)
from src.train import _score_experiment as train_score_experiment
from src.utils import DataContractError


def _base_rows(issue: str, n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "issue": issue,
                "candidate_number": i,
                "retrieval_next_draw_posterior": 0.1,
                "cand_hits_last_100": float(i % 10),
                "cand_current_gap": float((i % 8) + 1),
                "cand_pm1_neighbor_hits": float(i % 5),
                "cand_pm2_neighbor_hits": float(i % 3),
                "cand_rebound_score": float(i % 7) / 10.0,
                "cand_recent_reactivation_score": float(i % 9) / 10.0,
                "retrieval_similarity_max": 0.2,
                "retrieval_similarity_mean": 0.2,
                "retrieval_exact_draw_match_count_mean": 1.0,
                "retrieval_dynamic_context_n": 20.0,
                "retrieval_same_day_progress_bonus": 0.1,
                "label": 1 if i <= 20 else 0,
            }
        )
    return pd.DataFrame(rows)


def _dynamic_cfg(enabled: bool = True) -> DynamicWeightConfig:
    return DynamicWeightConfig.from_mapping(
        {
            "enabled": enabled,
            "mode": "heuristic_retrieval_gate_v1",
            "min_weights": {"ranker": 0.35, "logistic": 0.08, "retrieval": 0.10},
            "fixed_weights": {"history_prior": 0.10, "analysis": 0.03, "local_peak": 0.02},
        }
    )


def test_dynamic_weighting_moves_with_retrieval_quality() -> None:
    base = RuntimeWeights(0.4, 0.15, 0.3, 0.1, 0.03, 0.02)
    cfg = _dynamic_cfg(True)
    strong = _base_rows("I1")
    weak = _base_rows("I2")

    strong["retrieval_similarity_max"] = 0.95
    strong["retrieval_similarity_mean"] = 0.90
    strong["retrieval_next_draw_posterior"] = 0.85
    strong["retrieval_exact_draw_match_count_mean"] = 18.0
    strong["retrieval_same_day_progress_bonus"] = 1.0

    weak["retrieval_similarity_max"] = 0.05
    weak["retrieval_similarity_mean"] = 0.05
    weak["retrieval_next_draw_posterior"] = 0.02
    weak["retrieval_exact_draw_match_count_mean"] = 0.1
    weak["retrieval_same_day_progress_bonus"] = 0.0

    w_strong, g_strong = resolve_issue_dynamic_weights(strong, base, cfg)
    w_weak, g_weak = resolve_issue_dynamic_weights(weak, base, cfg)
    assert g_strong > g_weak
    assert w_strong["retrieval"] > w_weak["retrieval"]
    assert w_weak["ranker"] > w_strong["ranker"]


def test_effective_weights_sum_to_one() -> None:
    base = RuntimeWeights(0.4, 0.15, 0.3, 0.1, 0.03, 0.02)
    cfg = _dynamic_cfg(True)
    rows = _base_rows("I1")
    eff, _ = resolve_issue_dynamic_weights(rows, base, cfg)
    assert abs(sum(eff.values()) - 1.0) <= 1e-9


def test_dynamic_disabled_matches_fixed_linear_sum() -> None:
    base = RuntimeWeights(0.4, 0.15, 0.3, 0.1, 0.03, 0.02)
    cfg = _dynamic_cfg(False)
    rows = _base_rows("I1")
    ranker = np.linspace(0.1, 1.0, len(rows))
    logistic = np.linspace(0.2, 0.8, len(rows))
    scored = score_candidates(rows, ranker, logistic, base, dynamic_cfg=cfg)
    expected = (
        base.ranker * scored["ranker_score"]
        + base.logistic * scored["logistic_score"]
        + base.retrieval * scored["retrieval_score"]
        + base.history_prior * scored["history_prior_score"]
        + base.analysis * scored["analysis_rerank_score"]
        + base.local_peak * scored["local_peak_score"]
    )
    assert np.allclose(scored["final_score"].to_numpy(), expected.to_numpy())


def test_backtest_predict_shared_compose_parity() -> None:
    base = RuntimeWeights(0.4, 0.15, 0.3, 0.1, 0.03, 0.02)
    cfg = _dynamic_cfg(True)
    rows = _base_rows("I1")
    rows["ranker_score"] = np.linspace(0.1, 1.0, len(rows))
    rows["logistic_score"] = np.linspace(0.2, 0.8, len(rows))
    rows["retrieval_score"] = rows["retrieval_next_draw_posterior"]
    rows["history_prior_score"] = np.linspace(0.0, 1.0, len(rows))
    rows["analysis_rerank_score"] = np.linspace(1.0, 0.0, len(rows))
    rows["local_peak_score"] = np.linspace(0.3, 0.7, len(rows))
    a = compose_final_score_from_components(rows, base, cfg)
    b = compose_final_score_from_components(rows, base, cfg)
    assert np.allclose(a["final_score"].to_numpy(), b["final_score"].to_numpy())


def test_fail_fast_missing_dynamic_config_keys() -> None:
    with pytest.raises(DataContractError):
        DynamicWeightConfig.from_mapping({"enabled": True})


def test_fail_fast_missing_retrieval_columns() -> None:
    base = RuntimeWeights(0.4, 0.15, 0.3, 0.1, 0.03, 0.02)
    cfg = _dynamic_cfg(True)
    rows = _base_rows("I1").drop(columns=["retrieval_similarity_max"])
    with pytest.raises(DataContractError):
        resolve_issue_dynamic_weights(rows, base, cfg)


def test_fail_fast_dynamic_gate_nan() -> None:
    base = RuntimeWeights(0.4, 0.15, 0.3, 0.1, 0.03, 0.02)
    cfg = _dynamic_cfg(True)
    rows = _base_rows("I1")
    rows["retrieval_similarity_max"] = np.nan
    with pytest.raises(DataContractError):
        resolve_issue_dynamic_weights(rows, base, cfg)


def test_fail_fast_fixed_weights_invalid_sum() -> None:
    with pytest.raises(DataContractError):
        DynamicWeightConfig.from_mapping(
            {
                "enabled": True,
                "mode": "heuristic_retrieval_gate_v1",
                "min_weights": {"ranker": 0.35, "logistic": 0.08, "retrieval": 0.10},
                "fixed_weights": {"history_prior": 0.8, "analysis": 0.2, "local_peak": 0.2},
            }
        )


def test_train_ablation_no_retrieval_forces_dynamic_off() -> None:
    base = RuntimeWeights(0.4, 0.15, 0.3, 0.1, 0.03, 0.02)
    dyn_on = _dynamic_cfg(True)
    dyn_off = _dynamic_cfg(False)
    rows = _base_rows("I1")
    rows["ranker_score"] = np.linspace(0.1, 1.0, len(rows))
    rows["logistic_score"] = np.linspace(0.2, 0.8, len(rows))
    rows["retrieval_score"] = rows["retrieval_next_draw_posterior"]
    rows["history_prior_score"] = np.linspace(0.0, 1.0, len(rows))
    rows["analysis_rerank_score"] = np.linspace(1.0, 0.0, len(rows))
    rows["local_peak_score"] = np.linspace(0.3, 0.7, len(rows))

    scored_ablation = train_score_experiment(rows, "ablation_no_retrieval", base, dyn_on)
    expected = compose_final_score_from_components(
        rows,
        RuntimeWeights(
            ranker=base.ranker,
            logistic=base.logistic,
            retrieval=0.0,
            history_prior=base.history_prior,
            analysis=base.analysis,
            local_peak=base.local_peak,
        ),
        dyn_off,
    )
    assert np.allclose(scored_ablation["final_score"].to_numpy(), expected["final_score"].to_numpy())

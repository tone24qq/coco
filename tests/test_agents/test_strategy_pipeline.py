import numpy as np
import pandas as pd

from src.strategy import StrategyConfig, apply_strategy, issue_metrics


def test_rerank_changes_scores_shape_and_values() -> None:
    cand = pd.DataFrame(
        {
            "cand_freq_smooth_200": np.linspace(0, 1, 80),
            "cand_freq_trend_20_200": np.linspace(-0.2, 0.2, 80),
            "cand_recent_hit_decay_hl5": np.linspace(0, 2, 80),
            "cand_pmi_last_draw_sum_200": np.linspace(0, 1, 80),
            "num_zone": np.array([(i // 20) for i in range(80)], dtype=float),
        }
    )
    base = np.linspace(0.01, 0.99, 80)
    cfg = StrategyConfig("v3", "rerank", 30, 300, 3.0, 0.1, 0.3, True)
    out = apply_strategy(base, cand, cfg, "balanced")
    assert out.shape == base.shape
    assert float(np.abs(out - base).sum()) > 0


def test_issue_metrics_has_top3_at_least_one() -> None:
    scores = np.linspace(0.0, 1.0, 80)
    actual = {78, 79, 80}
    metric = issue_metrics(scores, actual)
    assert metric["top3_at_least_one_hit_rate"] == 1.0
    assert "top5_hit_rate" in metric
    assert "ndcg_at_10" in metric

import numpy as np
import pandas as pd

from src.strategy import StrategyConfig, apply_strategy, issue_metrics


def test_rerank_changes_scores_shape_and_values() -> None:
    cand = pd.DataFrame(
        {
            "freq_last_100": np.linspace(0, 5, 80),
            "freq_last_300": np.linspace(1, 6, 80),
            "freq_last_500": np.linspace(2, 7, 80),
            "ema_short_minus_ema_long": np.linspace(-1, 1, 80),
            "cooccur_with_last_draw_mean": np.linspace(0, 2, 80),
            "num_zone": np.array([(i // 20) for i in range(80)], dtype=float),
        }
    )
    base = np.linspace(0.01, 0.99, 80)
    cfg = StrategyConfig("v2", "rerank", 30, 300, 3.0, 0.1, 0.3, True)
    out = apply_strategy(base, cand, cfg, "balanced")
    assert out.shape == base.shape
    assert float(np.abs(out - base).sum()) > 0


def test_issue_metrics_has_top3_at_least_one() -> None:
    scores = np.linspace(0.0, 1.0, 80)
    actual = {78, 79, 80}
    metric = issue_metrics(scores, actual)
    assert metric["top3_at_least_one_hit_rate"] == 1.0

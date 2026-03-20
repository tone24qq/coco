import numpy as np
import pandas as pd

from src.runtime_scoring import RuntimeWeights, score_candidates


def test_runtime_score_formula_parity() -> None:
    df = pd.DataFrame(
        {
            "issue": ["I1"] * 3,
            "candidate_number": [1, 2, 3],
            "retrieval_next_draw_posterior": [0.2, 0.1, 0.0],
            "cand_hits_last_100": [5.0, 4.0, 3.0],
            "cand_current_gap": [2.0, 5.0, 8.0],
            "cand_pm1_neighbor_hits": [1.0, 2.0, 3.0],
            "cand_pm2_neighbor_hits": [0.0, 1.0, 2.0],
            "cand_rebound_score": [0.3, 0.2, 0.1],
            "cand_recent_reactivation_score": [0.4, 0.2, 0.1],
        }
    )
    ranker = np.array([0.3, 0.2, 0.1])
    logistic = np.array([0.2, 0.1, 0.05])
    w = RuntimeWeights(ranker=0.55, logistic=0.2, retrieval=0.15, history_prior=0.08, analysis=0.01, local_peak=0.01)
    scored = score_candidates(df, ranker, logistic, w)
    assert len(scored) == 3
    assert "analysis_rerank_score" in scored.columns

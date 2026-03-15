from __future__ import annotations

import numpy as np
import pandas as pd


def run_cascade_flow(
    issue_row: pd.Series,
    base_scores: np.ndarray,
    stage1_keep: int = 30,
    stage2_keep: int = 10,
) -> dict[str, object]:
    _ = issue_row, base_scores, stage1_keep, stage2_keep
    raise ValueError(
        "run_cascade_flow is deprecated in phase2; use CascadePipeline.predict_issue with stage artifacts"
    )

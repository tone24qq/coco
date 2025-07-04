from typing import Any, Dict, List, Optional

import numpy as np

import brain
from modules import fuse_scores
from weights import AGG_WEIGHTS


def predict_location(
    grid: List[List[int]], target: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Return ranked cell locations using fused module scores."""

    arr = np.asarray(grid, dtype=int)
    mods = list(brain.REGISTERED_MODULES_BRAIN)
    scores = {name: brain.get_module_score(name, arr, target=target) for name in mods}
    fused = fuse_scores(scores, arr, AGG_WEIGHTS)
    blanks = np.argwhere(arr == -1)
    preds = [
        {"row": int(r), "col": int(c), "score": float(fused[r, c])} for r, c in blanks
    ]
    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds

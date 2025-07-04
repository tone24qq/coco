from typing import Any, Dict, List, Optional

import numpy as np

import brain
from fusion import borda_rank, fuse_scores_dynamic
from modules import fuse_scores


def predict_location(
    grid: List[List[int]],
    target: Optional[int] = None,
    *,
    rank_method: str = "linear",
) -> List[Dict[str, Any]]:
    """Return ranked cell locations using fused module scores.

    Parameters
    ----------
    rank_method:
        "linear" for static weights,
        "dynamic" for adaptive weighting,
        "borda" for Borda count ranking.
    """

    arr = np.asarray(grid, dtype=int)
    mods = list(brain.REGISTERED_MODULES_BRAIN)
    scores = {name: brain.get_module_score(name, arr, target=target) for name in mods}

    if rank_method == "dynamic":
        fused = np.zeros_like(arr, dtype=float)
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                cell = {m: float(scores[m][r, c]) for m in mods}
                fused[r, c] = fuse_scores_dynamic(cell)
    elif rank_method == "borda":
        mp = {
            m: {
                (r, c): float(scores[m][r, c])
                for r in range(arr.shape[0])
                for c in range(arr.shape[1])
            }
            for m in mods
        }
        ranked = borda_rank(mp, top_n=arr.size)
        fused = np.zeros_like(arr, dtype=float)
        for (r, c), sc in ranked:
            fused[r, c] = sc
        if fused.max() > 0:
            fused = fused / float(fused.max())
    else:
        fused = fuse_scores(scores, arr, brain.AGG_WEIGHTS)
    blanks = np.argwhere(arr == -1)
    preds = [
        {"row": int(r), "col": int(c), "score": float(fused[r, c])} for r, c in blanks
    ]
    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds

from __future__ import annotations

from typing import Dict

import numpy as np


Array2D = np.ndarray


def normalize_scores(scores: Array2D, missing_mask: Array2D) -> Array2D:
    out = np.zeros_like(scores, dtype=float)
    vals = scores[missing_mask]
    if vals.size == 0:
        return out
    mn = float(np.min(vals))
    mx = float(np.max(vals))
    if np.isclose(mn, mx):
        out[missing_mask] = 1.0
    else:
        out[missing_mask] = (vals - mn) / (mx - mn)
    return out


def fuse_scores(module_scores: Dict[str, Array2D], weights: Dict[str, float], missing_mask: Array2D) -> Array2D:
    fused = np.zeros_like(next(iter(module_scores.values())), dtype=float)
    for name, score in module_scores.items():
        w = float(weights.get(name, 0.0))
        fused += w * score
    fused[~missing_mask] = 0.0
    return normalize_scores(fused, missing_mask)

"""Score fusion utilities."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Base weights for dynamic weighting
BASE_WEIGHTS: Dict[str, float] = {
    "conn": 0.4,
    "focus": 0.3,
    "tail": 0.01,
    "diff": 0.0,
    "mirror": 0.0,
}
# - 可用環境變數 FUSION_CONN_BOOST 調整 +0.20 的增幅
# - 若未來新增模組（如 "entropy"），只需在 BASE_WEIGHTS 補 key，
#   未補 key 會自動 fallback = 0.0，不會 KeyError。


def fuse_scores_dynamic(
    scores: Dict[str, float | None],
    base: Dict[str, float] | None = None,
) -> float:
    """Fuse module scores with adaptive weighting."""
    if base is None:
        base = BASE_WEIGHTS

    valid = {k: v for k, v in scores.items() if v and v > 1e-6}
    if not valid:
        valid = {"conn": scores.get("conn", 0.0)}

    # 容錯：缺失 key -> 0.0
    weights = {k: base.get(k, 0.0) for k in valid}
    if (
        valid.get("conn", 0.0) > 0.9
        and sum(v for k, v in valid.items() if k != "conn") < 0.3
    ):
        boost = float(os.getenv("FUSION_CONN_BOOST", "0.2"))
        conn_score = valid.get("conn", 0.0)
        return min(conn_score + boost, 1.0)

    total_w = sum(weights.values()) or 1.0
    weights = {k: w / total_w for k, w in weights.items()}

    return sum(valid[k] * weights.get(k, 0.0) for k in valid)


# ========= Vec-speed up (可選) =========
def fuse_scores_dynamic_grid(score_stack: np.ndarray, modules: List[str]) -> np.ndarray:
    """Vectorized version handling entire grid.

    Parameters
    ----------
    score_stack:
        Array of shape (M, H, W) where M=len(modules).
    modules:
        Module name ordering corresponding to score_stack.
    """
    out = np.zeros_like(score_stack[0], dtype=float)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            cell = {m: float(score_stack[i, r, c]) for i, m in enumerate(modules)}
            out[r, c] = fuse_scores_dynamic(cell)
    return out


# =======================================


def borda_rank(
    module_maps: Dict[str, Dict[Tuple[int, int], float]],
    top_n: int = 10,
) -> List[Tuple[Tuple[int, int], int]]:
    """Return Borda-count ranked cells."""
    borda: Dict[Tuple[int, int], int] = defaultdict(int)
    for mp in module_maps.values():
        ranked: List[Tuple[float, Tuple[int, int]]] = sorted(
            [(v, xy) for xy, v in mp.items() if v is not None],
            reverse=True,
        )
        n = len(ranked)
        for rank, (_, xy) in enumerate(ranked, 1):
            borda[xy] += n - rank
    return sorted(borda.items(), key=lambda x: -x[1])[:top_n]

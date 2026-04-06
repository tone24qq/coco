from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .fusion import fuse_scores
from .modules import compute_module_score


Cell = Tuple[int, int]


@dataclass
class TargetPrediction:
    board_id: str
    repeat_id: int
    target_row: int
    target_col: int
    true_value: int
    rank: int
    num_candidates: int
    top1_hit: int
    top3_hit: int
    top5_hit: int
    ranking_score: float


def legal_candidates(masked_grid: np.ndarray) -> List[int]:
    visible = set(int(v) for v in masked_grid[masked_grid != -1].tolist())
    return sorted(list(set(range(1, 81)) - visible))


def score_candidate(
    masked_grid: np.ndarray,
    target_cell: Cell,
    candidate_value: int,
    heatmap_prior: np.ndarray | None,
    modules: List[str],
) -> Dict[str, float]:
    r, c = target_cell
    if masked_grid[r, c] != -1:
        raise ValueError("target_cell must be masked")
    candidate_grid = masked_grid.copy()
    candidate_grid[r, c] = candidate_value
    return {
        module: compute_module_score(
            module,
            masked_grid,
            target_cell,
            candidate_value,
            candidate_grid,
            heatmap_prior,
        )
        for module in modules
    }


def rank_candidates(
    masked_grid: np.ndarray,
    target_cell: Cell,
    true_value: int,
    weights: Dict[str, float],
    heatmap_prior: np.ndarray | None,
    modules: List[str],
) -> Tuple[int, float]:
    candidates = legal_candidates(masked_grid)
    scores: List[Tuple[int, float]] = []
    for cand in candidates:
        feats = score_candidate(masked_grid, target_cell, cand, heatmap_prior, modules)
        scores.append((cand, fuse_scores(feats, weights)))
    scores.sort(key=lambda x: x[1], reverse=True)
    rank = next(i for i, (cand, _) in enumerate(scores, start=1) if cand == true_value)
    score_true = next(sc for cand, sc in scores if cand == true_value)
    return rank, float(score_true)


def random_rank(masked_grid: np.ndarray, true_value: int, rng: np.random.Generator) -> int:
    cands = legal_candidates(masked_grid)
    perm = list(cands)
    rng.shuffle(perm)
    return perm.index(true_value) + 1

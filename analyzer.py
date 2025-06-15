# analyzer.py

import os
import logging
import math
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Any
from functools import lru_cache
from modules import FORMULA_REGISTRY, compute_global_features, AdaptiveWeights
from brain import EXT_GM20_Skip_Pattern_Confidence_Vec, MathUtils, BoardAnalyzerUtils

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def simulate_with_formulas(
    grid_bytes: bytes,
    rows: int,
    cols: int,
    n_iter: int,
    weights: Dict[str, float] = None
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """
    Monte Carlo simulation to estimate probabilities for blank cells.
    """
    math_utils = MathUtils()
    analyzer = BoardAnalyzerUtils()
    rng = np.random.default_rng()
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)

    batch_size = 10000 if rows * cols < 50 else 5000 if rows * cols < 200 else 1000
    logger.debug(f"simulate_with_formulas: rows={rows}, cols={cols}, n_iter={n_iter}, batch_size={batch_size}")

    blanks = np.argwhere(grid == -1)
    known_idx = np.argwhere(grid != -1)
    known_vals = grid[grid != -1]
    hit_counter = {tuple(b): Counter() for b in map(tuple, blanks)}

    w = weights or {"excel": 0.6, "shuffle": 0.4}
    names = list(w)
    lin_known = rows * known_idx[:, 0] + known_idx[:, 1]
    skip_scores = EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    mean_val, std_val = compute_global_features(grid.astype(np.float32))

    full_batches = max(1, n_iter // batch_size)
    leftover = n_iter % batch_size

    for batch_idx in range(full_batches + (1 if leftover else 0)):
        current_size = batch_size if batch_idx < full_batches else leftover
        if current_size == 0:
            continue

        boards = np.zeros((current_size, rows * cols), dtype=np.int64)
        choices = rng.choice(names, size=current_size, p=[w[n] for n in names])
        for i, fname in enumerate(choices):
            boards[i] = FORMULA_REGISTRY[fname](rows, cols, rng).ravel()

        valid = np.all(boards[:, lin_known] == known_vals, axis=1)
        if not valid.any():
            continue
        valid_boards = boards[valid].reshape(-1, rows, cols)

        seq_mask = np.array([analyzer.check_sequences(b) for b in valid_boards])
        valid_boards = valid_boards[seq_mask]

        if valid_boards.size:
            board_scores = np.array([EXT_GM20_Skip_Pattern_Confidence_Vec(b) for b in valid_boards])
            corr_mask = np.array([
                np.corrcoef(skip_scores.ravel(), bs.ravel())[0, 1] > 0.8
                for bs in board_scores
            ])
            valid_boards = valid_boards[corr_mask]
            board_scores = board_scores[corr_mask]

        for b_idx, board in enumerate(valid_boards):
            for (r, c) in blanks:
                val = int(board[r, c])
                window = board[max(0, r-1):r+2, max(0, c-1):c+2]
                known_neigh = window[window != -1]
                resonance = 1.0
                if known_neigh.size:
                    resonance = 1 / (1 + abs(val - known_neigh.mean()))
                global_weight = math.exp(-((val - mean_val)**2) / (2 * std_val**2))
                score = board_scores[b_idx, r, c] * resonance * global_weight
                hit_counter[(r, c)][val] += score

        if all(
            cnt and max(cnt.values()) / sum(cnt.values()) > 0.95
            for cnt in hit_counter.values()
        ):
            logger.debug(f"simulate_with_formulas: early stop at batch {batch_idx}")
            break

    prob_map: Dict[Tuple[int, int], Dict[int, float]] = {}
    for pos, cnt in hit_counter.items():
        total = sum(cnt.values()) or 1.0
        prob_map[pos] = {
            num: math_utils.normalize_value(v, 0, total)
            for num, v in cnt.items()
        }

    logger.debug(f"simulate_with_formulas: completed with cells {list(prob_map.keys())}")
    return prob_map

def weight_prob_by_modules(
    grid: np.ndarray,
    prob_map: Dict[Tuple[int, int], Dict[int, float]]
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """
    Apply heuristic modules to re-weight probabilities:
    local resonance, skip-pattern, sequence boost, global gaussian.
    """
    math_utils = MathUtils()
    analyzer = BoardAnalyzerUtils()
    rows, cols = grid.shape
    blanks = np.argwhere(grid == -1)

    logger.debug("weight_prob_by_modules: start")

    # Local resonance
    for r, c in blanks:
        window = grid[max(0, r-1):r+2, max(0, c-1):c+2]
        known_vals = window[window != -1]
        if known_vals.size:
            for num in list(prob_map[(r, c)].keys()):
                prob_map[(r, c)][num] *= 1 / (1 + abs(num - known_vals.mean()))

    # Skip-pattern
    skip_scores = EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    for r, c in blanks:
        for num in list(prob_map[(r, c)].keys()):
            prob_map[(r, c)][num] *= skip_scores[r, c]

    # Sequence boost
    for r, c in blanks:
        row_seqs = analyzer.get_arithmetic_or_geometric_sequences(grid[r], min_len=3, allow_gaps=1)
        col_seqs = analyzer.get_arithmetic_or_geometric_sequences(grid[:, c], min_len=3, allow_gaps=1)
        for num in list(prob_map[(r, c)].keys()):
            if any(num in seq for seq in row_seqs + col_seqs):
                prob_map[(r, c)][num] *= 1.5

    # Global gaussian
    mean_val, std_val = compute_global_features(grid.astype(np.float32))
    for r, c in blanks:
        for num in list(prob_map[(r, c)].keys()):
            w = math.exp(-((num - mean_val)**2) / (2 * std_val**2))
            prob_map[(r, c)][num] *= w

    # Normalize final probabilities
    for pos in prob_map:
        total = sum(prob_map[pos].values()) or 1.0
        prob_map[pos] = {
            k: math_utils.normalize_value(v, 0, total)
            for k, v in prob_map[pos].items()
        }

    logger.debug("weight_prob_by_modules: done")
    return prob_map

def predict_scratch_card(
    grid: List[List[int]],
    n_iter: int,
    use_formula_only: bool = False
) -> Dict[str, Any]:
    """
    Predict top-3 candidates for each blank in the grid.
    """
    import os
    grid_np = np.array(grid, dtype=np.int64)
    logger.debug(f"predict_scratch_card: n_iter={n_iter}, formula_only={use_formula_only}")
    prob_map = simulate_with_formulas(
        grid_np.tobytes(), grid_np.shape[0], grid_np.shape[1], n_iter
    )

    if use_formula_only or os.getenv("USE_FORMULA_ONLY") == "1":
        logger.info("predict_scratch_card: formula-only mode, skipping module weighting")
    else:
        prob_map = weight_prob_by_modules(grid_np, prob_map)

    results: List[Dict[str, Any]] = []
    for (r, c), dist in prob_map.items():
        top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        nums, confs = zip(*top3) if top3 else ((), ())
        results.append({
            "row": int(r),
            "col": int(c),
            "candidates": list(nums),
            "confidences": [round(v, 4) for v in confs]
        })

    sorted_results = sorted(
        results,
        key=lambda x: x["confidences"][0] if x["confidences"] else 0.0,
        reverse=True
    )
    full_probs = {f"{r},{c}": dist for (r, c), dist in prob_map.items()}

    logger.debug("predict_scratch_card: done")
    return {"predictions": sorted_results, "full_probabilities": full_probs}
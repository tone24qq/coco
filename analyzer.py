import numpy as np
import logging
from modules import ScratchSolver
import asyncio
import json
import os
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_board(
    grid: np.ndarray,
    weights: Dict[str, float],
    return_predictions: bool = False,
    target_num: int = None,
    json_heatmap_path: str = None,
    knowledge_base: List[Dict[str, Any]] = None,
    heatmap_data: Dict[str, Any] = None
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, float, Dict[str, float]]], Dict[str, float]]:
    """
    Analyzes a scratch card board to predict hidden numbers.

    Returns:
        final_score: ndarray of fused scores for each hidden cell
        predictions: ndarray of predicted values (same shape as grid, -1 for hidden)
        top3: list of (row, col, score, module_contributions)
        metrics: dict of evaluation metrics
    """
    # --- 0. load KB & heatmaps logs ---
    if knowledge_base:
        logger.info(f"Loaded {len(knowledge_base)} KB concepts")
    if heatmap_data:
        logger.info(f"Loaded {len(heatmap_data)} heatmap files")

    # --- 1. validate grid & uniqueness ---
    solver = ScratchSolver()
    solver.update_tree(grid)

    # support only 4x4 to 20x20
    h, w = grid.shape
    if not (4 <= h <= 20 and 4 <= w <= 20):
        logger.error("Grid size out of 4x4 to 20x20 bounds")
        return np.array([]), grid.copy(), [], {"accuracy":0, "pattern_match":0, "value_diff":0}

    N = h * w
    opened = grid[grid != -1].tolist()
    if len(opened) != len(set(opened)) or any(n < 1 or n > N for n in opened):
        logger.error("Numbers violate 1~N uniqueness rule")
        return np.array([]), grid.copy(), [], {"accuracy":0, "pattern_match":0, "value_diff":0}

    # if user specified target_num, block duplicates
    if target_num is not None and target_num in opened:
        logger.warning(f"Target number {target_num} already present on board")
        return np.array([]), grid.copy(), [], {"accuracy":0, "pattern_match":0, "value_diff":0}

    # prepare set of used numbers
    used_numbers = set(opened)

    # --- 2. load JSON heatmap if path given ---
    initial_scores = None
    if json_heatmap_path and os.path.exists(json_heatmap_path):
        try:
            with open(json_heatmap_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            hm = np.array(data.get('heatmap', []), dtype=float)
            if hm.shape == grid.shape:
                initial_scores = hm
                logger.info(f"Successfully loaded heatmap from {json_heatmap_path}")
            else:
                logger.warning(f"Heatmap shape {hm.shape} != grid shape {grid.shape}")
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read JSON heatmap: {e}")
    else:
        logger.info(f"No valid heatmap at {json_heatmap_path}, skip")

    # --- 3. module score computation ---
    mod_scores: Dict[str, np.ndarray] = {}
    empties = np.argwhere(grid == -1)
    empty_count = len(empties)
    for mod_name, mod_func in solver.MODULE_REGISTRY.items():
        try:
            res = mod_func(grid)
            arr = res[0] if isinstance(res, tuple) else res
            if arr.size != empty_count:
                logger.warning(f"{mod_name}: expected {empty_count} scores but got {arr.size}")
                arr = np.zeros(empty_count)
        except Exception as e:
            logger.error(f"{mod_name} failed: {e}")
            arr = np.zeros(empty_count)
        mod_scores[mod_name] = arr

    # --- 4. fuse module scores ---
    board_type = solver.classify_board_type(mod_scores.get('compute_dynamic_hot_cold_vectorized', np.zeros(empty_count)))
    solver.adaptive_weights.update(success_rate=0.0, module_scores=mod_scores)
    final_score = solver.fuse_scores_vectorized(mod_scores, board_type, solver.adaptive_weights.weights)

    # if initial_scores exists, blend lightly (lambda_heat small)
    if initial_scores is not None:
        # map heatmap to a flat array matching empties order
        heat_vals = np.array([ initial_scores[r, c] for r, c in empties ], dtype=float)
        # normalize both to [0,1]
        def norm(x):
            amin, amax = x.min(), x.max()
            return (x - amin)/(amax - amin + 1e-9)
        f_norm = norm(final_score)
        h_norm = norm(heat_vals)
        λ = 0.1
        final_score = f_norm * (1-λ) + h_norm * λ

    # --- 5. predict patterns & integrate ---
    patterns = solver.analyze_number_patterns(grid)
    predictions, confidence = solver.integrate_predictions(grid, final_score, patterns)

    # --- 6. build candidates, filter duplicates ---
    preds = []
    for idx, (r, c) in enumerate(empties):
        pred_val = predictions[r, c] if return_predictions else None
        # if predicting a specific target, only keep that
        if target_num is not None:
            # skip all other positions
            if pred_val != target_num:
                continue
        # filter out any duplicate predictions
        if pred_val is not None and pred_val in used_numbers:
            continue
        preds.append((r, c, final_score[idx], {mod: mod_scores[mod][idx] for mod in mod_scores}))

    # ensure at least three candidates
    if not preds:
        return final_score, predictions, [], {"accuracy":0, "pattern_match":0, "value_diff":0}
    preds = sorted(preds, key=lambda x: x[2], reverse=True)
    top3 = preds[:3] + preds[-0:]  # ensure length

    # --- 7. evaluate (optional) ---
    # create a dummy true_values matrix for metrics
    all_nums = set(range(1, N+1))
    remaining = list(all_nums - used_numbers)
    np.random.shuffle(remaining)
    true_vals = grid.copy()
    for (r, c), num in zip(empties, remaining):
        true_vals[r, c] = num
    metrics = solver.evaluate_prediction(grid, predictions, true_vals)

    return final_score, predictions, top3, metrics
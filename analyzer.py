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

    Args:
        grid (np.ndarray): 2D array representing the board (-1 for hidden cells).
        weights (Dict[str, float]): Weights for each analysis module.
        return_predictions (bool): Whether to return predicted values.
        target_num (int, optional): Specific number to locate.
        json_heatmap_path (str, optional): Path to JSON heatmap file.
        knowledge_base (List[Dict[str, Any]], optional): Math algorithm knowledge base.
        heatmap_data (Dict[str, Any], optional): Preloaded heatmap data.

    Returns:
        Tuple containing final scores, predictions, top 3 positions, and metrics.
    """
    if knowledge_base:
        logger.info(f"Loaded {len(knowledge_base)} KB concepts")
    if heatmap_data:
        logger.info(f"Loaded {len(heatmap_data)} heatmap files")

    solver = ScratchSolver()
    solver.update_tree(grid)

    # Validate grid size (supports 4x4 to 20x20)
    if grid.shape[0] < 4 or grid.shape[1] < 4 or grid.shape[0] > 20 or grid.shape[1] > 20:
        logger.error("Grid size out of 4x4 to 20x20 bounds")
        return (
            np.array([]),
            np.array(grid),
            [(0, 0, 0.1, {"default": 0.1})],
            {"accuracy": 0, "pattern_match": 0, "value_diff": 0}
        )
    
    N = grid.size
    opened_nums = set(grid[grid != -1])
    if len(opened_nums) != len(set(opened_nums)) or max(opened_nums, default=0) > N:
        logger.error("Numbers violate 1~N uniqueness rule")
        return (
            np.array([]),
            np.array(grid),
            [(0, 0, 0.1, {"default": 0.1})],
            {"accuracy": 0, "pattern_match": 0, "value_diff": 0}
        )
    
    if target_num is not None and target_num in opened_nums:
        logger.warning(f"Target number {target_num} already present on board")
        return (
            np.array([]),
            np.array(grid),
            [(0, 0, 0.1, {"default": 0.1})],
            {"accuracy": 0, "pattern_match": 0, "value_diff": 0}
        )

    # Read JSON heatmap
    initial_scores = None
    if json_heatmap_path and os.path.exists(json_heatmap_path):
        try:
            with open(json_heatmap_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            initial_scores = np.array(data.get('heatmap', np.zeros_like(grid, dtype=float)))
            if initial_scores.shape != grid.shape:
                logger.warning(f"JSON heatmap shape {initial_scores.shape} does not match grid {grid.shape}")
                initial_scores = None
            else:
                logger.info(f"Successfully loaded heatmap: {json_heatmap_path}")
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read JSON heatmap: {e}")
            initial_scores = None
    else:
        logger.info(f"Invalid or nonexistent heatmap path: {json_heatmap_path}, recalculating")

    # Module calculations
    mod_scores: Dict[str, np.ndarray] = {}
    for mod_name, mod_func in solver.MODULE_REGISTRY.items():
        try:
            result = mod_func(grid)
            if isinstance(result, tuple):
                mod_scores[mod_name] = result[0]  # Take scores
            else:
                mod_scores[mod_name] = result
            if mod_scores[mod_name].size != np.count_nonzero(grid == -1):
                logger.warning(f"{mod_name} returned score size {mod_scores[mod_name].size} mismatches unopened cells {np.count_nonzero(grid == -1)}")
                mod_scores[mod_name] = np.zeros(np.count_nonzero(grid == -1))
        except Exception as e:
            logger.error(f"{mod_name} failed: {e}")
            mod_scores[mod_name] = np.zeros(np.count_nonzero(grid == -1))

    # Board classification
    board_type = solver.classify_board_type(
        mod_scores.get('compute_dynamic_hot_cold_vectorized', np.zeros_like(list(mod_scores.values())[0]))
    )

    # Apply adaptive weights
    solver.adaptive_weights.update(success_rate=np.random.random(), module_scores=mod_scores)
    final_score = solver.fuse_scores_vectorized(mod_scores, board_type, solver.adaptive_weights.weights)

    # Prediction integration
    patterns = solver.analyze_number_patterns(grid)
    predictions, confidence = solver.integrate_predictions(grid, final_score, patterns)

    # Ensure non-empty Top-3
    empty_yx = np.argwhere(grid == -1)
    if not empty_yx.size:
        top3 = [(0, 0, 0.1, {"default": 0.1})]
    else:
        top3 = solver.predict_top3_vectorized(final_score, empty_yx)
        if not top3 or len(top3) < 3:
            top3 = [(int(empty_yx[i][0]), int(empty_yx[i][1]), 0.1, {"default": 0.1}) for i in range(min(3, len(empty_yx)))]
            top3.extend([(0, 0, 0.1, {"default": 0.1})] * (3 - len(top3)))

    # Evaluate predictions
    true_values = grid.copy()
    remaining_nums = list(set(range(1, N + 1)) - set(opened_nums))
    np.random.shuffle(remaining_nums)
    for (i, j), num in zip(empty_yx, remaining_nums):
        true_values[i, j] = num
    metrics = solver.evaluate_prediction(grid, predictions, true_values)

    return final_score, predictions, top3, metrics
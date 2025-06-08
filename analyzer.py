# analyzer.py (continued in script.py)
import numpy as np
import logging
from modules import ScratchSolver

# 設置日誌
logger = logging.getLogger(__name__)

def analyze_board(grid: np.ndarray, weights: dict, return_predictions: bool = False, target_num: int = None) -> tuple:
    """
    Analyze the scratch card grid to predict hidden numbers.

    Args:
        grid (np.ndarray): 2D array representing the scratch card grid.
        weights (dict): Weights for different analysis modules.
        return_predictions (bool): Whether to return predictions.
        target_num (int): Specific number to locate (optional).

    Returns:
        tuple: (score array, prediction array, best positions list) or appropriate subsets.

    Raises:
        ValueError: If grid validation fails or target number is invalid.
    """
    solver = ScratchSolver()
    solver.update_tree(grid)

    # Validate grid size and rules
    if grid.shape[0] > 20 or grid.shape[1] > 20 or grid.shape[0] < 4 or grid.shape[1] < 5:
        logger.error("Grid size must be between 4x5 and 20x20")
        raise ValueError("Grid size must be between 4x5 and 20x20")
    N = grid.size
    opened_nums = set(grid[grid != -1])
    if len(opened_nums) != len(set(opened_nums)) or max(opened_nums, default=0) > N:
        logger.error("Numbers must be unique from 1 to N")
        raise ValueError("Numbers must be unique from 1 to N")

    if target_num is not None and target_num in opened_nums:
        logger.error(f"Target number {target_num} already opened")
        return None, None, {"error": f"Target number {target_num} already opened"}

    # Compute module scores
    mod_scores = {}
    try:
        mod_scores['compute_dynamic_hot_cold_vectorized'] = solver.compute_dynamic_hot_cold_vectorized(grid)
    except Exception as e:
        logger.error(f"Module compute_dynamic_hot_cold_vectorized failed: {e}")
        mod_scores['compute_dynamic_hot_cold_vectorized'] = np.zeros(np.count_nonzero(grid == -1))

    try:
        mod_scores['compute_block_heatmap_vectorized'] = solver.compute_block_heatmap_vectorized(grid)
    except Exception as e:
        logger.error(f"Module compute_block_heatmap_vectorized failed: {e}")
        mod_scores['compute_block_heatmap_vectorized'] = np.zeros(np.count_nonzero(grid == -1))

    # Classify board type and fuse scores
    board_type = solver.classify_board_type(mod_scores['compute_dynamic_hot_cold_vectorized'])
    final_score = solver.fuse_scores_vectorized(mod_scores, board_type, weights)

    # Prepare empty positions
    empty_yx = np.argwhere(grid == -1)

    if return_predictions:
        top3 = solver.predict_top3_vectorized(final_score, empty_yx)
        return final_score, None, top3
    else:
        return final_score, None, None
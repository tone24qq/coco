# ------------------- dependencies -------------------
# pip install fastapi uvicorn ortools tabulate numpy

import json, os, time, logging, uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool # Not explicitly used in new logic, but good for general FastAPI
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Tuple, Callable, Any, Optional
import numpy as np
from ortools.sat.python import cp_model
from tabulate import tabulate

# --- Logging設定 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)

# --- 路徑設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEM_PATH = os.path.join(BASE_DIR, "memory_cards.json") # Not used in this snippet
REASONING_LOG_PATH = os.path.join(BASE_DIR, "reasoning_log.jsonl") # Not used in this snippet
MODULE_WEIGHTS_PATH = os.path.join(BASE_DIR, "module_weights.json") # Not used, weights hardcoded

# --- 表格輸出工具 ---
def format_data_as_table(data_to_format: Any, headers_option: Any = None, tablefmt: str = "grid", floatfmt: str = ".2f", generate_default_headers_if_numpy_2d_and_no_headers: bool = False) -> str:
    current_headers = headers_option
    if isinstance(data_to_format, np.ndarray):
        final_data = data_to_format.tolist()
        if generate_default_headers_if_numpy_2d_and_no_headers and \
           (headers_option is None or headers_option == []) and \
           data_to_format.ndim == 2:
            num_cols = data_to_format.shape[1]
            current_headers = [f"Col {i+1}" for i in range(num_cols)]
    elif isinstance(data_to_format, list):
        # Ensure list of lists for tabulate
        if data_to_format and not isinstance(data_to_format[0], list):
            final_data = [data_to_format] # Wrap if it's a flat list representing a single row
        else:
            final_data = data_to_format
    else:
        logger.warning(f"Unsupported data type for table formatting: {type(data_to_format)}")
        return f"Unsupported data type for table formatting: {type(data_to_format)}"

    if not final_data or (isinstance(final_data, list) and all(not row for row in final_data)):
        return "No data to format."
    
    actual_tabulate_headers = current_headers if current_headers is not None else []
    
    try:
        return tabulate(final_data, headers=actual_tabulate_headers, tablefmt=tablefmt, floatfmt=floatfmt if floatfmt else ".2f")
    except Exception as e:
        logger.error(f"Error during table formatting: {e}", exc_info=True)
        return f"Error formatting table: {str(e)}"

app = FastAPI(title="MetaCognitive Scratch Card Solver (v5.1-Operational)", version="5.1")

# --- Heuristic Functions (死邏輯全集 - Your Original Functions) ---
# These functions calculate scores for cells based on various patterns and features.
# -1 in the grid represents an empty/unknown cell.
# Scores are generally higher for cells that are "more interesting" by the heuristic's criteria.

def a2_center_radial_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    center = np.array([(H - 1) / 2, (W - 1) / 2])
    score_map = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == -1:
                dist = np.sqrt((r - center[0])**2 + (c - center[1])**2)
                norm_dist = dist / (np.sqrt((H/2)**2 + (W/2)**2) or 1) # Normalize by max possible distance from center
                score_map[r, c] = 1.0 - norm_dist
    return score_map

def a5_adj_density_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    padded_grid_is_filled = np.pad(grid != -1, ((1, 1), (1, 1)), 'constant', constant_values=False)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == -1:
                # Count filled neighbors in the original grid
                filled_neighbors = 0
                # Corresponding indices in padded grid are (r+1, c+1)
                # Check N, S, E, W
                if padded_grid_is_filled[r, c+1]: filled_neighbors +=1 # North (original r-1, c)
                if padded_grid_is_filled[r+2, c+1]: filled_neighbors +=1 # South (original r+1, c)
                if padded_grid_is_filled[r+1, c]: filled_neighbors +=1 # West (original r, c-1)
                if padded_grid_is_filled[r+1, c+2]: filled_neighbors +=1 # East (original r, c+1)
                score_map[r, c] = filled_neighbors / 4.0 # Normalize by max neighbors
    return score_map

def a6_fixed_position_vec(grid: np.ndarray) -> np.ndarray:
    # This heuristic gives a base score to all empty cells.
    return (grid == -1).astype(float)

def a8_symmetry_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    flipped_grid = np.fliplr(grid) # Flip horizontally
    for r in range(H):
        for c in range(W):
            if grid[r, c] == -1: # Only score empty cells
                # If the symmetric cell in the flipped grid is also empty, it's a candidate for symmetric placement
                # A simple score could be if the values *would be* symmetric if chosen.
                # For positional symmetry, if grid[r,c] is empty, check grid[r, W-1-c]
                # If grid[r, W-1-c] is also -1, it's perfectly symmetric potential.
                # If grid[r, W-1-c] has a value, less symmetric.
                # This heuristic as originally written compares grid == flip, which would be 0 if grid[r,c] is -1.
                # Let's interpret it as: how well would *filling this cell* maintain symmetry?
                # A simpler positional score:
                if grid[r, W-1-c] == -1 : # If symmetric cell is also empty
                     score_map[r,c] = 1.0
                elif grid[r,c] == flipped_grid[r,c]: # This would be if grid[r,c] is NOT -1
                     pass # Original logic was (grid == flip).astype(float) * mask
    # A more direct interpretation for empty cells:
    # Score is higher if the symmetrically opposite cell is also empty (potential for symmetric fill)
    # or if filling it would match a filled symmetric cell (less likely for scratch cards).
    # Let's stick to a simple interpretation: score if the cell itself *could* be part of a symmetric pattern.
    # The original logic `(grid == flip).astype(float) * mask` means:
    # if cell (r,c) is -1 (due to mask), AND grid[r,c] == flipped_grid[r,c] (i.e. -1 == grid[r, W-1-c])
    # then score is 1. Otherwise 0.
    mask = (grid == -1)
    score = (grid == np.fliplr(grid)).astype(float) # Where values are identical after flip
    return score * mask # Apply only to empty cells that are symmetric to their counterpart


def m1_uni_gap_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        row_vals = grid[i, :]
        filled_indices = np.where(row_vals != -1)[0]
        if len(filled_indices) > 1:
            gaps = np.diff(filled_indices)
            if len(gaps) > 0: # Need at least one gap to calculate std
                 # Normalize std: max std could be (W-1)/2 if alternating filled/empty.
                 # A simpler normalization: stddev relative to mean gap or W.
                mean_gap = np.mean(gaps) if len(gaps) > 0 else W # Avoid division by zero if no gaps
                s = 1.0 - (np.std(gaps) / (mean_gap if mean_gap > 0 else (W or 1)))
                s = max(0,s) # ensure score is not negative
                # Apply this score to all empty cells in that row
                for c_idx in range(W):
                    if grid[i, c_idx] == -1:
                        score[i, c_idx] = s
            elif len(gaps) == 0 and len(filled_indices) > 1: # e.g. [1,2,-1,-1] -> filled_indices=[0,1], diff=[1], std=0
                for c_idx in range(W):
                    if grid[i, c_idx] == -1:
                        score[i,c_idx] = 1.0 # Perfect uniformity if only one gap or adjacent items
    return score # Already implicitly masked by how it's applied or use `* mask` if needed

def m2_seq_pattern_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    score = np.zeros_like(grid, dtype=float)
    # Score for rows
    for r in range(H):
        row_values = grid[r, grid[r, :] != -1]
        if len(row_values) > 2: # Need at least 3 values to see a pattern in differences
            sorted_row_values = np.sort(row_values)
            diffs = np.diff(sorted_row_values)
            if len(diffs) > 1: # Need at least 2 differences to check their std
                # If std of diffs is small, it's like an arithmetic progression
                # Normalize std of diffs, e.g., by mean of diffs or a constant
                # Original: (np.std(d) < 2) * 1.0. This is a binary score.
                # Let's make it continuous: 1 / (1 + std(diffs))
                std_dev_diffs = np.std(diffs)
                current_row_score = 1.0 / (1.0 + std_dev_diffs)
                for c in range(W):
                    if grid[r,c] == -1:
                        score[r,c] += current_row_score # Add to score for empty cells in this row
    # Score for columns
    for c in range(W):
        col_values = grid[grid[:, c] != -1, c]
        if len(col_values) > 2:
            sorted_col_values = np.sort(col_values)
            diffs = np.diff(sorted_col_values)
            if len(diffs) > 1:
                std_dev_diffs = np.std(diffs)
                current_col_score = 1.0 / (1.0 + std_dev_diffs)
                for r in range(H):
                    if grid[r,c] == -1:
                        score[r,c] += current_col_score # Add to score for empty cells in this col
    return score * mask # Ensure only empty cells get scores

def m3_diff_band_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == -1: # Only for empty cells
                neighbor_values = []
                # Check N, S, E, W neighbors
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != -1:
                        # Original: abs(grid[ni,nj]-(grid[i,j] if grid[i,j]!=-1 else 0))
                        # This implies comparing with 0 if the current cell is -1.
                        # Let's assume we want to see if a potential fill would match a band.
                        # For now, just average neighbor values.
                        # The original logic seems to be: for an empty cell, what are the diffs of its neighbors to 0?
                        # This interpretation is odd.
                        # Let's assume it means: if this cell were 0, what's the average diff to its neighbours?
                        # vals.append(abs(grid[nr,nc] - 0)) -> vals.append(abs(grid[nr,nc]))
                        neighbor_values.append(abs(grid[nr, nc]))
                
                if neighbor_values:
                    mean_diff_to_zero_of_neighbors = np.mean(neighbor_values)
                    # Original: 1.0 if 5 <= s <= 20 else 0.3
                    # This is a score for the cell, not a value to put in it.
                    if 5 <= mean_diff_to_zero_of_neighbors <= 20:
                        score_map[r, c] = 1.0
                    else:
                        score_map[r, c] = 0.3
    return score_map

def m4_biaxial_stat_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score_map = (grid == -1).astype(float) # Start with base score of 1 for empty cells
    
    if W > 0:
        row_density = np.sum(grid != -1, axis=1) / W
    else:
        row_density = np.zeros(H)
        
    if H > 0:
        col_density = np.sum(grid != -1, axis=0) / H
    else:
        col_density = np.zeros(W)

    for r in range(H):
        for c in range(W):
            if grid[r, c] == -1: # Only for empty cells
                # Original: mask[i,j] *= (0.5 < row_density[i] < 0.8) and (0.5 < col_density[j] < 0.8)
                # This means the score is 1 if conditions met, 0 otherwise.
                if not (0.5 < row_density[r] < 0.8 and 0.5 < col_density[c] < 0.8):
                    score_map[r,c] = 0.0 # Penalize if densities are outside desired range
    return score_map

def m5_bar_focus_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    score = np.zeros_like(grid, dtype=float)
    # Score for rows being dense
    if W > 0 : # avoid division by zero
        for r in range(H):
            if np.sum(grid[r,:] != -1) > W // 2:
                score[r,:] += 1 # Add score to all cells in that row (will be masked later)
    # Score for columns being dense
    if H > 0 : # avoid division by zero
        for c in range(W):
            if np.sum(grid[:,c] != -1) > H // 2:
                score[:,c] += 1 # Add score to all cells in that col (will be masked later)
    return score * mask # Max score 2 for a cell if both its row and col are dense

def m6_neighbor_cycle_vec(grid: np.ndarray) -> np.ndarray: # Renamed from cycle to density (more accurate)
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == -1: # Only for empty cells
                filled_neighbors = 0
                num_potential_neighbors = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]: # N, S, W, E
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        num_potential_neighbors +=1
                        if grid[nr, nc] != -1:
                            filled_neighbors += 1
                if num_potential_neighbors > 0:
                    score_map[r, c] = filled_neighbors / num_potential_neighbors
                else: # Should not happen in a grid > 1x1
                    score_map[r, c] = 0
    return score_map


def m7_bisec_zone_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    score_val = 0.0
    if H > 0 and W > 0 :
        row_fill_counts = np.array([np.sum(grid[i,:] != -1) for i in range(H)])
        col_fill_counts = np.array([np.sum(grid[:,j] != -1) for j in range(W)])
        
        # Normalize std dev by mean or total possible items
        # Low std dev means uniform distribution of items per row/col
        std_row_norm = np.std(row_fill_counts) / (W or 1)
        std_col_norm = np.std(col_fill_counts) / (H or 1)
        
        # Average normalized std dev. Lower is better. Score is inverse.
        # Original: (s < 1.5)*1.0 where s was (np.std(row_chunks) + np.std(col_chunks)) / (H+W)
        # Let's use a continuous score:
        combined_std_metric = (std_row_norm + std_col_norm) / 2.0
        score_val = 1.0 / (1.0 + combined_std_metric) # Higher score for lower std_metric
    
    score_map = np.full_like(grid, score_val, dtype=float)
    return score_map * mask

def m8_repeat_gap_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        row_line = grid[i,:]
        filled_indices = np.where(row_line != -1)[0]
        if len(filled_indices) > 2: # Need at least 3 filled items to have 2 gaps
            gaps = np.diff(filled_indices)
            if len(gaps) > 1: # Need at least 2 gaps
                std_gaps = np.std(gaps)
                mean_gaps = np.mean(gaps)
                # Original: (s < avg) * 1.0
                # Score higher if std of gaps is small compared to mean gap (more regular)
                if mean_gaps > 0: # Avoid division by zero
                    current_row_score = 1.0 - (std_gaps / mean_gaps) # Higher is better, max 1
                    current_row_score = max(0, current_row_score) # Ensure non-negative
                else: # If mean_gaps is 0, implies all filled cells are adjacent, so std_gaps is 0. Perfect regularity.
                    current_row_score = 1.0

                for c_idx in range(W):
                    if grid[i, c_idx] == -1:
                        score[i,c_idx] = current_row_score
    return score * mask


def m9_double_rule_overlap_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    score = np.zeros_like(grid, dtype=float)
    # This heuristic scores based on existing pairs. It should score empty cells
    # based on their potential to form such pairs.
    # For an empty cell (r,c), check its neighbors (r,c-1) and (r,c+1).
    # If grid[r,c-1] is filled, and grid[r,c+1] is filled, this cell is between them.
    # This isn't what the original seems to do. Original: score[i,j] += ... for filled grid[i,j]
    # Let's adapt: an empty cell gets a score if placing a number there *could* complete such a pattern.
    for r in range(H):
        for c in range(W):
            if grid[r,c] == -1:
                # Check left neighbor
                if c > 0 and grid[r,c-1] != -1:
                    # If a value here made |val - grid[r,c-1]| == 5 or 10
                    # This is value-dependent. Positional score: potential.
                    score[r,c] += 0.5 # Potential to form pair with left
                # Check right neighbor
                if c < W - 1 and grid[r,c+1] != -1:
                    score[r,c] += 0.5 # Potential to form pair with right
    return score * mask


def m10_seq_order_match_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        row_values = grid[i, grid[i,:] != -1]
        if len(row_values) > 1:
            is_increasing = np.all(np.diff(row_values) > 0)
            is_decreasing = np.all(np.diff(row_values) < 0)
            if is_increasing or is_decreasing:
                # Apply score to all empty cells in this row
                for c_idx in range(W):
                    if grid[i,c_idx] == -1:
                        score[i,c_idx] = 1.0
    return score * mask # Apply to empty cells

def m11_block_match_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    # Determine a reasonable block size, e.g., 2x2 or 3x3, but not too large.
    # Original: max(min(H,W)//4, 2). Let's make it adaptive or fixed for simplicity e.g. 2 or 3.
    block_h = max(H // 3, 2) if H > 1 else 1
    block_w = max(W // 3, 2) if W > 1 else 1
    
    score_map = np.zeros_like(grid, dtype=float)

    for r_start in range(0, H - block_h + 1, block_h): # Iterate with step_size = block_size
        for c_start in range(0, W - block_w + 1, block_w):
            block = grid[r_start : r_start+block_h, c_start : c_start+block_w]
            num_filled_in_block = np.sum(block != -1)
            total_cells_in_block = block.size
            
            if total_cells_in_block > 0:
                density = num_filled_in_block / total_cells_in_block
                # Original: if nonempty > (block_size*block_size)//2: score += 1
                # Apply density score to empty cells within this block
                for r_in_block in range(block_h):
                    for c_in_block in range(block_w):
                        actual_r, actual_c = r_start + r_in_block, c_start + c_in_block
                        if grid[actual_r, actual_c] == -1: # If the cell in original grid is empty
                            score_map[actual_r, actual_c] = density # Score is the density of its block
    return score_map * mask


def f2_row_rotate_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    score = np.zeros_like(grid, dtype=float)
    # This implies a connection between end of one row and start of next.
    # An empty cell gets a score if it could participate in such a rotation.
    # E.g., if grid[i-1, W-1] is filled and grid[i,0] is empty, then grid[i,0] is a candidate.
    for r in range(1, H): # Starts from the second row
        if W > 0 and grid[r-1, W-1] != -1: # If end of previous row is filled
             if grid[r,0] == -1: # And start of current row is empty
                score[r,0] += 1.0
    return score * mask

def f3_col_rotate_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    score = np.zeros_like(grid, dtype=float)
    # Connection between end of one col and start of next.
    for c in range(1, W): # Starts from the second col
        if H > 0 and grid[H-1, c-1] != -1: # If end of previous col is filled
            if grid[0,c] == -1: # And start of current col is empty
                score[0,c] += 1.0
    return score * mask

def r2_rev_diff_vec(grid: np.ndarray) -> np.ndarray: # Similar to m10 but specifically for decreasing
    H, W = grid.shape
    mask = (grid == -1)
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        row_values = grid[i, grid[i,:] != -1]
        if len(row_values) > 1:
            if np.all(np.diff(row_values) < 0): # Strictly decreasing
                for c_idx in range(W):
                    if grid[i, c_idx] == -1:
                        score[i,c_idx] = 1.0
    return score * mask

def r7_odd_even_dist_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid == -1)
    score_val = 0.0
    
    filled_cells = grid[grid != -1]
    if filled_cells.size > 0:
        num_odd = np.sum((filled_cells % 2 != 0)) # Consider 0 as even
        num_even = np.sum((filled_cells % 2 == 0))
        total_filled = len(filled_cells)
        
        if total_filled > 0:
            # Ratio of imbalance: abs_diff / total. Lower is better (more balanced).
            imbalance_ratio = abs(num_odd - num_even) / total_filled
            # Score is 1 - imbalance_ratio. Higher score for more balance.
            # Original: (ratio < 0.3)*1.0
            score_val = 1.0 - imbalance_ratio if imbalance_ratio < 0.3 else 0.1
    else: # No filled cells, perfect balance by default? Or neutral score.
        score_val = 0.5 

    score_map = np.full_like(grid, score_val, dtype=float)
    return score_map * mask

def d3_pair_freq_vec(grid: np.ndarray) -> np.ndarray: # Similar to m9
    H, W = grid.shape
    mask = (grid == -1)
    score = np.zeros_like(grid, dtype=float)
    # Score empty cells based on potential to form pairs with specific differences
    for r in range(H):
        for c in range(W):
            if grid[r,c] == -1:
                # Check with left neighbor grid[r, c-1]
                if c > 0 and grid[r,c-1] != -1:
                    # If placing a value V here, would abs(V - grid[r,c-1]) be in [1,9,10]?
                    # This is value-dependent. Positional score: potential.
                    score[r,c] += 0.5
                # Check with right neighbor grid[r, c+1]
                if c < W-1 and grid[r,c+1] != -1:
                    score[r,c] += 0.5
    return score * mask


# --- 集合全部死邏輯模組 ---
MODULE_FUNCS_VEC = {
    "A2": a2_center_radial_vec, "A5": a5_adj_density_vec, "A6": a6_fixed_position_vec, "A8": a8_symmetry_vec,
    "M1": m1_uni_gap_vec, "M2": m2_seq_pattern_vec, "M3": m3_diff_band_vec, "M4": m4_biaxial_stat_vec,
    "M5": m5_bar_focus_vec, "M6": m6_neighbor_cycle_vec, "M7": m7_bisec_zone_vec, "M8": m8_repeat_gap_vec,
    "M9": m9_double_rule_overlap_vec, "M10": m10_seq_order_match_vec, "M11": m11_block_match_vec,
    "F2": f2_row_rotate_vec, "F3": f3_col_rotate_vec,
    "R2": r2_rev_diff_vec, "R7": r7_odd_even_dist_vec,
    "D3": d3_pair_freq_vec,
}
MODULE_WEIGHTS = { # Default weights, can be loaded from file
    "A2": 0.7, "A5": 0.8, "A6": 0.6, "A8": 0.5, "M1": 0.6, "M2": 0.8, "M3": 0.9, "M4": 0.5,
    "M5": 0.5, "M6": 0.5, "M7": 0.5, "M8": 0.5, "M9": 0.6, "M10": 0.5, "M11": 0.5,
    "F2": 0.5, "F3": 0.5, "R2": 0.5, "R7": 0.5, "D3": 0.7,
}
# Attempt to load weights from file, otherwise use hardcoded
if os.path.exists(MODULE_WEIGHTS_PATH):
    try:
        with open(MODULE_WEIGHTS_PATH, 'r') as f:
            loaded_weights = json.load(f)
            # Validate loaded weights if necessary
            MODULE_WEIGHTS.update(loaded_weights) # Update defaults with loaded values
            logger.info(f"Successfully loaded module weights from {MODULE_WEIGHTS_PATH}")
    except Exception as e:
        logger.error(f"Error loading module weights from {MODULE_WEIGHTS_PATH}: {e}. Using default weights.")
else:
    logger.info(f"Module weights file not found at {MODULE_WEIGHTS_PATH}. Using default weights.")


# --- 張量分數 (Tensor Flow Score) ---
def tensor_flow_score_vec_all(grid: np.ndarray) -> np.ndarray:
    """Computes a weighted sum of all heuristic scores for each cell."""
    if grid.ndim != 2:
        raise ValueError("Input grid must be 2-dimensional.")
    if grid.size == 0:
        return np.array([[]], dtype=float) # Return empty 2D array matching shape convention

    total_score_map = np.zeros(grid.shape, dtype=float)
    active_heuristics_count = 0
    for name, func in MODULE_FUNCS_VEC.items():
        weight = MODULE_WEIGHTS.get(name, 0)
        if weight == 0:
            logger.debug(f"Skipping heuristic {name} due to zero weight.")
            continue
        
        try:
            heuristic_score_map = func(grid.copy()) # Pass a copy to avoid in-place modifications by heuristics
            if heuristic_score_map.shape != grid.shape:
                logger.error(f"Heuristic {name} returned map with shape {heuristic_score_map.shape}, expected {grid.shape}. Skipping.")
                continue
            total_score_map += heuristic_score_map.astype(float) * weight
            active_heuristics_count +=1
        except Exception as e:
            logger.error(f"Error executing heuristic {name}: {e}", exc_info=True)
            # Optionally, re-raise or handle more gracefully
    
    # Normalize if desired, e.g., by number of active heuristics or sum of weights
    # For now, it's a raw weighted sum.
    # Ensure scores are only for empty cells if not already handled by individual heuristics
    mask_empty = (grid == -1)
    return total_score_map * mask_empty

# --- Pydantic Models for API ---
class GridInput(BaseModel):
    grid: List[List[int]] = Field(..., example=[[-1, 10, -1], [5, -1, 20], [-1, 15, -1]])
    num_to_place: int = Field(default=1, gt=0, description="Number of cells to fill in this step.")
    value_domain_min: int = Field(default=1, description="Minimum possible value for a cell.")
    value_domain_max: int = Field(default=20, description="Maximum possible value for a cell.")

    @validator('grid')
    def check_grid_not_empty_and_rectangular(cls, v):
        if not v:
            raise ValueError("Grid cannot be empty.")
        if not isinstance(v[0], list):
            raise ValueError("Grid must be a list of lists.")
        if not v[0]: # First row empty
             raise ValueError("Grid rows cannot be empty.")
        row_len = len(v[0])
        if not all(len(row) == row_len for row in v):
            raise ValueError("Grid must be rectangular.")
        return v
    
    @validator('value_domain_max')
    def check_domain_max_ge_min(cls, v, values):
        if 'value_domain_min' in values and v < values['value_domain_min']:
            raise ValueError("value_domain_max must be greater than or equal to value_domain_min.")
        return v

class SolveStepResponse(BaseModel):
    new_grid: List[List[int]]
    chosen_cells: List[Tuple[int, int, int]] # List of (row, col, value) for cells that were filled
    solver_log: str
    status: str # CP-SAT solver status (OPTIMAL, FEASIBLE, INFEASIBLE, etc.)
    computed_scores_table: Optional[str] = None # Formatted table of scores for chosen cells or all empty

# --- Operational Logic using CP-SAT Solver ---
async def solve_step_cp(grid_input: GridInput) -> SolveStepResponse:
    """
    Uses CP-SAT to select optimal cells to fill based on heuristic scores.
    """
    current_grid_np = np.array(grid_input.grid)
    num_to_place = grid_input.num_to_place
    value_domain = (grid_input.value_domain_min, grid_input.value_domain_max)

    H, W = current_grid_np.shape
    empty_cells_coords_tuples = list(zip(*np.where(current_grid_np == -1))) # List of (r,c) tuples

    if not empty_cells_coords_tuples:
        return SolveStepResponse(
            new_grid=current_grid_np.tolist(), 
            chosen_cells=[], 
            solver_log="No empty cells to fill.", 
            status="NO_EMPTY_CELLS"
        )

    if num_to_place > len(empty_cells_coords_tuples):
        logger.warning(f"Requested to place {num_to_place} values, but only {len(empty_cells_coords_tuples)} empty cells. Adjusted to {len(empty_cells_coords_tuples)}.")
        num_to_place = len(empty_cells_coords_tuples)
    
    if num_to_place == 0:
         return SolveStepResponse(
             new_grid=current_grid_np.tolist(), 
             chosen_cells=[], 
             solver_log="Number of cells to place is 0. No action taken.", 
             status="NO_ACTION_REQUESTED"
            )

    # Calculate heuristic scores for all empty cells
    # Run tensor_flow_score_vec_all in a threadpool as it can be CPU intensive
    heuristic_scores_map = await run_in_threadpool(tensor_flow_score_vec_all, current_grid_np.copy())
    
    scores_at_empty_cells = [heuristic_scores_map[r, c] for r, c in empty_cells_coords_tuples]

    # CP-SAT prefers integer objectives. Scale scores.
    # Max possible raw score sum of weights ~10-15. Scale by 1000.
    scaling_factor = 1000.0
    scores_at_empty_cells_scaled = [int(s * scaling_factor) for s in scores_at_empty_cells]
    
    # Table for scores of empty cells (for debugging/logging)
    empty_cells_scores_data = []
    for i, (r,c) in enumerate(empty_cells_coords_tuples):
        empty_cells_scores_data.append([r, c, scores_at_empty_cells[i], scores_at_empty_cells_scaled[i]])
    scores_table_str = format_data_as_table(
        empty_cells_scores_data, 
        headers_option=["Row", "Col", "Raw Score", f"Scaled Score (x{int(scaling_factor)})"],
        tablefmt="pipe"
    )


    model = cp_model.CpModel()

    # Variables:
    # 1. `chosen_cell_indices`: For each of the `num_to_place` items, this var holds the index into `empty_cells_coords_tuples`.
    #    These indices must be distinct.
    chosen_cell_indices = [
        model.NewIntVar(0, len(empty_cells_coords_tuples) - 1, f"chosen_cell_idx_{i}")
        for i in range(num_to_place)
    ]
    if num_to_place > 0: # AddAllDifferent requires at least one variable
        model.AddAllDifferent(chosen_cell_indices)

    # 2. `assigned_values`: For each of the `num_to_place` items, this var holds the actual numeric value assigned.
    #    These values must be distinct.
    assigned_values = [
        model.NewIntVar(value_domain[0], value_domain[1], f"assigned_value_{i}")
        for i in range(num_to_place)
    ]
    if num_to_place > 0:
         model.AddAllDifferent(assigned_values)


    # Objective: Maximize the sum of scaled heuristic scores for the chosen cells.
    # For each `chosen_cell_indices[i]`, its score is `scores_at_empty_cells_scaled[chosen_cell_indices[i]]`.
    objective_terms = []
    if num_to_place > 0 : # AddElement requires a list and an index var.
        min_score_scaled = min(scores_at_empty_cells_scaled) if scores_at_empty_cells_scaled else 0
        max_score_scaled = max(scores_at_empty_cells_scaled) if scores_at_empty_cells_scaled else 0
        for i in range(num_to_place):
            term = model.NewIntVar(min_score_scaled, max_score_scaled, f"score_term_{i}")
            model.AddElement(chosen_cell_indices[i], scores_at_empty_cells_scaled, term)
            objective_terms.append(term)
        model.Maximize(sum(objective_terms))
    
    # Solve
    solver = cp_model.CpSolver()
    # solver.parameters.log_search_progress = True # Enable for detailed logs
    solver.parameters.max_time_in_seconds = 10.0 # Timeout for the solver

    status = solver.Solve(model)

    # Process results
    new_grid_np = current_grid_np.copy()
    chosen_cell_details_list = []
    log_message = f"CP-SAT Solver Status: {solver.StatusName(status)}\n"
    if objective_terms :
        log_message += f"Objective Value (Scaled): {solver.ObjectiveValue()}, Original Sum: {solver.ObjectiveValue()/scaling_factor}\n"
    else:
        log_message += "Objective Value: N/A (no terms)\n"
    log_message += f"Wall Time: {solver.WallTime()}s\n"

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i in range(num_to_place):
            original_empty_cell_idx = solver.Value(chosen_cell_indices[i])
            r, c = empty_cells_coords_tuples[original_empty_cell_idx]
            value_to_place = solver.Value(assigned_values[i])
            
            new_grid_np[r, c] = value_to_place
            chosen_cell_details_list.append( (int(r), int(c), int(value_to_place)) ) # Ensure native Python ints
            log_message += (
                f"  Decision {i+1}: Placed {value_to_place} at ({r},{c}). "
                f"(Original empty cell index: {original_empty_cell_idx}, "
                f"Score: {scores_at_empty_cells[original_empty_cell_idx]:.3f})\n"
            )
    else:
        log_message += "No solution found or problem was infeasible/aborted.\n"
        
    # Append scores table to log
    log_message += "\nScores for all considered empty cells:\n" + scores_table_str

    return SolveStepResponse(
        new_grid=new_grid_np.tolist(),
        chosen_cells=chosen_cell_details_list,
        solver_log=log_message,
        status=solver.StatusName(status),
        computed_scores_table=scores_table_str
    )

# --- FastAPI Endpoints ---
@app.post("/analyze_scores", summary="Get heuristic scores for the current grid (no solving)")
async def analyze_scores_endpoint(grid_input: GridInput):
    current_grid_np = np.array(grid_input.grid)
    
    # Run tensor_flow_score_vec_all in a threadpool
    heuristic_scores_map = await run_in_threadpool(tensor_flow_score_vec_all, current_grid_np.copy())
    
    empty_cells_coords_tuples = list(zip(*np.where(current_grid_np == -1)))
    scores_data = []
    for r,c in empty_cells_coords_tuples:
        scores_data.append([r, c, heuristic_scores_map[r,c]])
    
    scores_table_str = format_data_as_table(
        scores_data, 
        headers_option=["Row", "Col", "Combined Score"],
        tablefmt="pipe"
    )
    return {"message": "Heuristic scores computed.", "scores_table": scores_table_str, "raw_score_map": heuristic_scores_map.tolist()}

@app.post("/solve_step", response_model=SolveStepResponse, summary="Perform one step of solving using CP-SAT")
async def api_solve_step(grid_input: GridInput):
    """
    Takes the current grid, number of cells to place, and value domain.
    Returns the new grid after placing numbers in the heuristically best
    available empty cells, ensuring placed values are distinct.
    """
    try:
        response = await solve_step_cp(grid_input)
        return response
    except Exception as e:
        logger.error(f"Error in /solve_step: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Root endpoint with basic info")
async def read_root():
    return {"message": "MetaCognitive Scratch Card Solver API is running. Use POST /solve_step to interact."}

# --- Example Usage (for local testing if not running with uvicorn) ---
if __name__ == "__main__":
    # This part is for direct script execution testing, not part of the FastAPI app normally.
    # To run the FastAPI app: uvicorn your_filename:app --reload

    logger.info("Starting local test example...")
    example_grid_data = {
        "grid": [
            [-1, 12, -1,  8],
            [ 5, -1, 19, -1],
            [-1,  3, -1, 16],
            [10, -1,  7, -1]
        ],
        "num_to_place": 2,
        "value_domain_min": 1,
        "value_domain_max": 20
    }
    
    test_grid_input = GridInput(**example_grid_data)
    
    # Test score analysis
    logger.info("\n--- Testing Score Analysis ---")
    current_grid_np_test = np.array(test_grid_input.grid)
    heuristic_scores_map_test = tensor_flow_score_vec_all(current_grid_np_test.copy())
    empty_cells_coords_tuples_test = list(zip(*np.where(current_grid_np_test == -1)))
    scores_data_test = []
    for r_test,c_test in empty_cells_coords_tuples_test:
        scores_data_test.append([r_test, c_test, heuristic_scores_map_test[r_test,c_test]])
    scores_table_str_test = format_data_as_table(
        scores_data_test, 
        headers_option=["Row", "Col", "Combined Score"],
        tablefmt="grid" # Use grid for console
    )
    logger.info("Heuristic scores for empty cells:\n" + scores_table_str_test)
    logger.info("Raw score map:\n" + format_data_as_table(heuristic_scores_map_test, tablefmt="grid"))


    # Test solving step (requires asyncio loop if solve_step_cp is async)
    import asyncio
    logger.info("\n--- Testing Solve Step (CP-SAT) ---")
    try:
        solve_response = asyncio.run(solve_step_cp(test_grid_input))
        logger.info(f"Solver Status: {solve_response.status}")
        logger.info(f"Chosen Cells: {solve_response.chosen_cells}")
        logger.info("Solver Log:\n" + solve_response.solver_log)
        logger.info("New Grid:\n" + format_data_as_table(solve_response.new_grid, tablefmt="grid"))

    except Exception as e:
        logger.error(f"Error during local test of solve_step_cp: {e}", exc_info=True)

    # To run the actual API server:
    # Save this file (e.g., as main.py) then run in your terminal:
    # uvicorn main:app --reload
    logger.info("\nTo run the API server: uvicorn <filename>:app --reload")


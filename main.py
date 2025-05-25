# ------------------- dependencies -------------------
# pip install fastapi uvicorn ortools tabulate numpy

import json
import os
import time
import logging
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Tuple, Callable, Any, Optional
import numpy as np
from ortools.sat.python import cp_model
from tabulate import tabulate
from collections import Counter

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

# --- File paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEM_PATH = os.path.join(BASE_DIR, "memory_cards.json")
REASONING_LOG_PATH = os.path.join(BASE_DIR, "reasoning_log.jsonl")
MODULE_WEIGHTS_PATH = os.path.join(BASE_DIR, "module_weights.json")

# --- Table formatting utility ---
def format_data_as_table(
    data_to_format: Any,
    headers_option: Any = None,
    tablefmt: str = "grid",
    floatfmt: str = ".2f",
    generate_default_headers_if_numpy_2d_and_no_headers: bool = False
) -> str:
    headers = headers_option
    if isinstance(data_to_format, np.ndarray):
        data = data_to_format.tolist()
        if generate_default_headers_if_numpy_2d_and_no_headers and headers in (None, []) and data_to_format.ndim == 2:
            cols = data_to_format.shape[1]
            headers = [f"Col {i+1}" for i in range(cols)]
    elif isinstance(data_to_format, list):
        data = data_to_format
    else:
        logger.warning(f"Unsupported data type for table formatting: {type(data_to_format)}")
        return "Unsupported data type for table formatting."
    if not data or (isinstance(data, list) and all(not row for row in data)):
        return "No data to format."
    actual_headers = headers if headers is not None else []
    try:
        return tabulate(data, headers=actual_headers, tablefmt=tablefmt, floatfmt=floatfmt)
    except Exception as e:
        logger.error(f"Error during table formatting: {e}", exc_info=True)
        return f"Error formatting table: {e}"

app = FastAPI(
    title="MetaCognitive Scratch Card Solver (Combined v1+v2+Enhancements)",
    version="1.1"
)

# -----------------------------------------------------------------------------
# 1. Memory module (from version1)
# -----------------------------------------------------------------------------
_memory: Dict[str, Dict[str, Any]] = {}

def _make_board_id(grid: np.ndarray) -> str:
    H, W = grid.shape
    empty_count = int(np.sum(grid == -1))
    
    # Use a hash of the filled part of the grid for more robust ID
    # To make it canonical, sort the filled values or use a hash of the tuple of tuples
    filled_part_tuple = tuple(map(tuple, grid.tolist())) # Convert to tuple of tuples for hashing
    grid_hash = hash(filled_part_tuple)
    
    return f"{H}x{W}_empty{empty_count}_hash{grid_hash}"


def _load_memory() -> None:
    global _memory
    if os.path.exists(MEM_PATH):
        try:
            with open(MEM_PATH, "r", encoding="utf-8") as f:
                _memory = json.load(f)
            logger.info(f"Loaded memory ({len(_memory)}) from {MEM_PATH}")
        except Exception as e:
            logger.error(f"Failed to load memory: {e}", exc_info=True)
            _memory = {}
    else:
        _memory = {}
        logger.info("No memory file found; starting fresh.")

def _save_memory() -> None:
    try:
        with open(MEM_PATH, "w", encoding="utf-8") as f:
            json.dump(_memory, f, indent=2, sort_keys=True)
        logger.info(f"Saved memory ({len(_memory)}) to {MEM_PATH}")
    except Exception as e:
        logger.error(f"Failed to save memory: {e}", exc_info=True)

def update_memory(grid: np.ndarray, r: int, c: int, v: int, score: float, success: bool) -> None:
    bid = _make_board_id(grid) # Use original grid state for board ID
    key = f"{r}_{c}_{v}" # Move: (row, col, value)
    
    if bid not in _memory:
        _memory[bid] = {}
    
    entry = _memory[bid].setdefault(key, {"count": 0, "total_score": 0.0, "success_count": 0})
    entry["count"] += 1
    entry["total_score"] += score # Score of the heuristic evaluation that led to this choice
    if success:
        entry["success_count"] += 1

def mem_score(grid_id: str, r: int, c: int, v: int) -> Tuple[float, int]:
    """Returns (average_score, count) from memory for a specific move on a board_id."""
    key = f"{r}_{c}_{v}"
    if grid_id in _memory and key in _memory[grid_id]:
        entry = _memory[grid_id][key]
        count = entry.get("count", 0)
        if count > 0:
            # Consider success rate in score, e.g., avg_heuristic_score * (success_count / count)
            success_rate = entry.get("success_count", 0) / count
            avg_heuristic_score = entry["total_score"] / count
            # Simple weighted score: could be more sophisticated
            return avg_heuristic_score * success_rate, count
    return 0.0, 0

_load_memory()

# -----------------------------------------------------------------------------
# 2. Meta-cognition log (from version1)
# -----------------------------------------------------------------------------
class MetaCognitionLog:
    def __init__(self, path: str):
        self.path = path
        self.buffer: List[Dict[str, Any]] = []

    def log_event(self, event: Dict[str, Any]):
        event["log_id"] = str(uuid.uuid4())
        event["timestamp"] = time.time()
        # sanitize
        for k, v in list(event.items()):
            if isinstance(v, np.integer):
                event[k] = int(v)
            elif isinstance(v, np.floating):
                event[k] = float(v)
            elif isinstance(v, np.ndarray):
                event[k] = v.tolist()
            elif isinstance(v, tuple):
                event[k] = list(v)
        self.buffer.append(event)

    def flush(self):
        if not self.buffer:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for ev in self.buffer:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            logger.info(f"Flushed {len(self.buffer)} events to {self.path}")
            self.buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush log: {e}", exc_info=True)

meta_logger = MetaCognitionLog(REASONING_LOG_PATH)

# -----------------------------------------------------------------------------
# 3. Module weights management (from version1)
# -----------------------------------------------------------------------------
MODULE_WEIGHTS: Dict[str, float] = {}

def _load_module_weights() -> None:
    global MODULE_WEIGHTS
    defaults = {
        "A2": 0.7, "A5": 0.8, "A6": 0.6, "A8": 0.5,
        "M1": 0.6, "M2": 0.8, "M3": 0.9, "M4": 0.5, "M5": 0.5,
        "M6": 0.5, "M7": 0.5, "M8": 0.5, "M9": 0.6, "M10": 0.5,
        "M11": 0.5, "F2": 0.5, "F3": 0.5, "R2": 0.5, "R7": 0.5,
        "D3": 0.7,
        # New Heuristics Weights
        "H_ARITHMETIC": 0.8, # Heuristic for Arithmetic Progression Potential
        "H_MEMORY": 1.0,     # Heuristic for Memory-Based Score
    }
    if os.path.exists(MODULE_WEIGHTS_PATH):
        try:
            with open(MODULE_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            MODULE_WEIGHTS = {**defaults, **loaded} # Loaded can override defaults
            # Ensure all default keys are present if not in loaded file
            for key, value in defaults.items():
                MODULE_WEIGHTS.setdefault(key, value)
            logger.info(f"Loaded module weights from {MODULE_WEIGHTS_PATH}")
        except Exception as e:
            logger.error(f"Error loading weights: {e}", exc_info=True)
            MODULE_WEIGHTS = defaults
    else:
        MODULE_WEIGHTS = defaults
    # Save (potentially updated with new defaults)
    _save_module_weights()


def _save_module_weights() -> None:
    try:
        with open(MODULE_WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(MODULE_WEIGHTS, f, indent=2, sort_keys=True)
        logger.info(f"Saved module weights to {MODULE_WEIGHTS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save module weights: {e}", exc_info=True)

_load_module_weights()

# -----------------------------------------------------------------------------
# 4. Heuristic functions (A/M/F/R/D series, vectorized where possible)
#    + New Value-Aware Heuristics
# -----------------------------------------------------------------------------
# Existing Heuristics (assuming they take only 'grid')
def a2_center_radial_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    center = np.array([(H-1)/2, (W-1)/2])
    dist = np.sqrt((np.arange(H)[:,None]-center[0])**2 + (np.arange(W)-center[1])**2)
    norm = np.max(dist) or 1
    score = 1 - dist/norm
    return score * (grid==-1)

def a5_adj_density_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    padded = np.pad(grid!=-1, ((1,1),(1,1)), 'constant')
    dens = (
        padded[:-2,1:-1] + padded[2:,1:-1] +
        padded[1:-1,:-2] + padded[1:-1,2:]
    ) / 4.0
    return dens * (grid==-1)

def a6_fixed_position_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    return (grid==-1).astype(float)

def a8_symmetry_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    flip = np.fliplr(grid)
    return ((grid==flip).astype(float)) * (grid==-1)

def m1_uni_gap_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        filled = np.where(grid[i]!=-1)[0]
        if len(filled)>1:
            gaps = np.diff(filled)
            if len(gaps) > 0:
                 mean_gap = np.mean(gaps)
                 s = 1 - np.std(gaps)/(mean_gap if mean_gap > 0 else (W or 1))
                 score[i,:] = max(0,s)
            else: # Only two filled cells, perfectly uniform
                 score[i,:] = 1.0
    return score * (grid==-1)

def m2_seq_pattern_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        vals = np.sort(grid[i][grid[i]!=-1])
        if len(vals)>2:
            d = np.diff(vals)
            if len(d) > 1 and np.std(d) is not None: # ensure std is calculable
                score[i,:] += 1.0/(1+np.std(d))
            elif len(d) == 1: # e.g. [1,2,3] -> diffs [1,1], std=0. or [1,2] -> diff [1]
                 score[i,:] += 1.0
    for j in range(W):
        vals = np.sort(grid[:,j][grid[:,j]!=-1])
        if len(vals)>2:
            d = np.diff(vals)
            if len(d) > 1 and np.std(d) is not None:
                score[:,j] += 1.0/(1+np.std(d))
            elif len(d) == 1:
                 score[:,j] += 1.0
    return score * (grid==-1)

def m3_diff_band_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        for j in range(W):
            if grid[i,j]==-1:
                vals=[]
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni,nj=i+di,j+dj
                    if 0<=ni<H and 0<=nj<W and grid[ni,nj]!=-1:
                        vals.append(abs(grid[ni,nj]))
                if vals:
                    m = np.mean(vals)
                    score[i,j] = 1.0 if 5<=m<=20 else 0.3
    return score # Already applied to (grid==-1)

def m4_biaxial_stat_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if W > 0 and H > 0 :
        row_d = np.sum(grid!=-1,axis=1)/ W
        col_d = np.sum(grid!=-1,axis=0)/ H
        mask = (grid==-1)
        for i in range(H):
            for j in range(W):
                if mask[i,j] and 0.5<row_d[i]<0.8 and 0.5<col_d[j]<0.8:
                    score[i,j]=1.0
    return score

def m5_bar_focus_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    mask = (grid==-1)
    score = np.zeros_like(grid, dtype=float)
    if W > 0:
        for i in range(H):
            if np.sum(grid[i]!=-1)>W//2:
                score[i,:]+=1
    if H > 0:
        for j in range(W):
            if np.sum(grid[:,j]!=-1)>H//2:
                score[:,j]+=1
    return score*mask

def m6_neighbor_cycle_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        for j in range(W):
            if grid[i,j]==-1:
                cnt=0; tot=0
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni,nj=i+di,j+dj
                    if 0<=ni<H and 0<=nj<W:
                        tot+=1
                        if grid[ni,nj]!=-1: cnt+=1
                score[i,j]=cnt/(tot or 1)
    return score # Already applied to (grid==-1)

def m7_bisec_zone_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    s = 0.5 # Default score
    if H > 0 and W > 0 and grid.size > 0:
        rows = np.array([np.sum(grid[i]!=-1) for i in range(H)])
        cols = np.array([np.sum(grid[:,j]!=-1) for j in range(W)])
        if H+W > 0: # Avoid division by zero if H or W is 0 but not both (covered by H>0 and W>0)
            metric = (np.std(rows)+np.std(cols))/(H+W) # Removed 'or 1' as H+W > 0 here
            s = 1.0/(1+metric)
    return np.full_like(grid, s, dtype=float)*(grid==-1)

def m8_repeat_gap_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        filled = np.where(grid[i]!=-1)[0]
        if len(filled)>2:
            gaps=np.diff(filled)
            if len(gaps) > 0 :
                mean_gaps = np.mean(gaps)
                s = 1 - np.std(gaps)/(mean_gaps if mean_gaps > 0 else 1) # Avoid div by zero
                score[i,:]=max(0,s)
            else: # only two filled items, so one gap, std is 0 if mean is not 0.
                score[i,:] = 1.0
    return score*(grid==-1)

def m9_double_rule_overlap_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        for j in range(W):
            if grid[i,j]==-1:
                # Potential to form pair with left
                if j>0 and grid[i,j-1]!=-1: score[i,j]+=0.5
                # Potential to form pair with right
                if j<W-1 and grid[i,j+1]!=-1: score[i,j]+=0.5
    return score # Already applied to (grid==-1)

def m10_seq_order_match_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        vals = grid[i][grid[i]!=-1]
        if len(vals)>1 and (np.all(np.diff(vals)>0) or np.all(np.diff(vals)<0)):
            score[i,:]=1.0
    return score*(grid==-1)

def m11_block_match_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if H > 0 and W > 0:
        bs = max(min(H,W)//4, 2) if min(H,W) >= 4 else (min(H,W) if min(H,W)>0 else 1)

        for i in range(0,H-bs+1,bs):
            for j in range(0,W-bs+1,bs):
                block = grid[i:i+bs,j:j+bs]
                if block.size > 0:
                    den = np.sum(block!=-1)/block.size
                    # Apply score to empty cells within this block
                    for r_in_block in range(bs):
                        for c_in_block in range(bs):
                             if i+r_in_block < H and j+c_in_block < W and grid[i+r_in_block, j+c_in_block] == -1:
                                score[i+r_in_block,j+c_in_block] = den
    return score*(grid==-1) # Ensure only empty cells get score

def f2_row_rotate_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if W > 0: # Need width for grid[i-1, W-1]
        for i in range(1,H):
            if grid[i-1,W-1]!=-1 and grid[i,0]==-1:
                score[i,0]=1.0
    return score*(grid==-1)

def f3_col_rotate_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if H > 0: # Need height for grid[H-1, j-1]
        for j in range(1,W):
            if grid[H-1,j-1]!=-1 and grid[0,j]==-1:
                score[0,j]=1.0
    return score*(grid==-1)

def r2_rev_diff_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        vals=grid[i][grid[i]!=-1]
        if len(vals)>1 and np.all(np.diff(vals)<0):
            score[i,:]=1.0
    return score*(grid==-1)

def r7_odd_even_dist_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    filled = grid[grid!=-1]
    s = 0.5 # Default score
    if filled.size>0:
        odd = np.sum(filled%2!=0) # Non-zero is odd, 0 is even
        even = np.sum(filled%2==0)
        total_filled = odd+even # Recalculate total_filled based on actual odd/even counts
        if total_filled > 0 :
            ratio = abs(odd-even)/total_filled
            s = 1.0-ratio if ratio<0.3 else 0.1
    return np.full_like(grid, s, dtype=float)*(grid==-1)

def d3_pair_freq_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        for j in range(W): # Iterate all cells for potential
            if grid[i,j]==-1:
                # Check right neighbor
                if j < W-1 and grid[i,j+1]!=-1 and abs(grid[i,j+1]) in [1,9,10]:
                    score[i,j]+=0.5 # Original was 1.0, using 0.5 to sum up potentials
                # Check left neighbor
                if j > 0 and grid[i,j-1]!=-1 and abs(grid[i,j-1]) in [1,9,10]:
                    score[i,j]+=0.5
    return score*(grid==-1)


# --- New Heuristic Functions ---
def h_arithmetic_progression_potential(grid: np.ndarray, value_domain_min: int, value_domain_max: int, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    
    for r_idx in range(H):
        for c_idx in range(W):
            if grid[r_idx, c_idx] == -1: # Only for empty cells
                max_progression_score = 0.0
                
                # Check horizontal and vertical potential for each candidate value
                for candidate_val in range(value_domain_min, value_domain_max + 1):
                    current_value_score = 0.0
                    # Horizontal check
                    left_vals = []
                    if c_idx > 0 and grid[r_idx, c_idx-1] != -1: left_vals.append(grid[r_idx, c_idx-1])
                    if c_idx > 1 and grid[r_idx, c_idx-2] != -1 and grid[r_idx, c_idx-1] == -1: # Can't form with this heuristic type
                        pass # Simplified: only immediate neighbors
                    
                    right_vals = []
                    if c_idx < W - 1 and grid[r_idx, c_idx+1] != -1: right_vals.append(grid[r_idx, c_idx+1])

                    # Case 1: candidate_val is between two numbers: L, X, R
                    if left_vals and right_vals:
                        l_val, r_val = left_vals[0], right_vals[0]
                        if candidate_val - l_val == r_val - candidate_val: # Forms AP
                            current_value_score += 1.0
                    # Case 2: X, R1, R2 (candidate is leftmost)
                    elif not left_vals and len(right_vals) > 0 and c_idx < W - 2 and grid[r_idx, c_idx+2] != -1:
                        r1_val, r2_val = right_vals[0], grid[r_idx, c_idx+2]
                        if r1_val - candidate_val == r2_val - r1_val:
                            current_value_score += 0.75 # Slightly less score for edge completion
                    # Case 3: L1, L2, X (candidate is rightmost)
                    elif not right_vals and len(left_vals) > 0 and c_idx > 1 and grid[r_idx, c_idx-2] != -1:
                        l1_val, l2_val = grid[r_idx, c_idx-2], left_vals[0]
                        if l2_val - l1_val == candidate_val - l2_val:
                            current_value_score += 0.75
                    # Case 4: L, X (candidate is right, needs one left) or X, R (candidate is left, needs one right)
                    # For simplicity, only 3-term progressions considered robustly. We can add score for 2-term potential.
                    # If L, X then X-L is a diff. If X, R then R-X is a diff.

                    # Vertical check (similar logic)
                    up_vals = []
                    if r_idx > 0 and grid[r_idx-1, c_idx] != -1: up_vals.append(grid[r_idx-1, c_idx])
                    down_vals = []
                    if r_idx < H - 1 and grid[r_idx+1, c_idx] != -1: down_vals.append(grid[r_idx+1, c_idx])

                    if up_vals and down_vals:
                        u_val, d_val = up_vals[0], down_vals[0]
                        if candidate_val - u_val == d_val - candidate_val:
                            current_value_score += 1.0
                    elif not up_vals and len(down_vals) > 0 and r_idx < H - 2 and grid[r_idx+2, c_idx] != -1:
                        d1_val, d2_val = down_vals[0], grid[r_idx+2, c_idx]
                        if d1_val - candidate_val == d2_val - d1_val:
                            current_value_score += 0.75
                    elif not down_vals and len(up_vals) > 0 and r_idx > 1 and grid[r_idx-2, c_idx] != -1:
                        u1_val, u2_val = grid[r_idx-2, c_idx], up_vals[0]
                        if u2_val - u1_val == candidate_val - u2_val:
                             current_value_score += 0.75
                    
                    if current_value_score > max_progression_score:
                        max_progression_score = current_value_score
                
                score_map[r_idx, c_idx] = max_progression_score # Assign best potential score
    return score_map * (grid==-1)


def h_memory_based_score(grid: np.ndarray, value_domain_min: int, value_domain_max: int, **kwargs) -> np.ndarray:
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    current_grid_id = _make_board_id(grid) # Get ID for current grid state
    
    num_possible_values = value_domain_max - value_domain_min + 1
    if num_possible_values <= 0: return score_map # Should not happen with valid domain

    for r_idx in range(H):
        for c_idx in range(W):
            if grid[r_idx, c_idx] == -1:
                total_mem_score_for_cell = 0.0
                total_counts_for_cell = 0
                max_mem_score_for_cell = 0.0
                
                for val_candidate in range(value_domain_min, value_domain_max + 1):
                    avg_score, count = mem_score(current_grid_id, r_idx, c_idx, val_candidate)
                    if count > 0:
                        total_mem_score_for_cell += avg_score * count # Weight by count
                        total_counts_for_cell += count
                        if avg_score > max_mem_score_for_cell:
                             max_mem_score_for_cell = avg_score
                
                # Aggregate: Use max score found for any value at this position, or weighted average
                # Using max_mem_score_for_cell gives a signal if any past value was very good.
                # Using weighted average is more robust if many values have some history.
                # Let's use max for now, normalized by typical heuristic score ranges (e.g. 0-2)
                # If heuristic scores leading to memory are ~0-10, and mem_score is avg_score*success_rate,
                # this could be directly used. Max score of 1 from heuristic * 1 success rate = 1.
                if total_counts_for_cell > 0:
                    # score_map[r_idx, c_idx] = total_mem_score_for_cell / total_counts_for_cell
                    score_map[r_idx, c_idx] = max_mem_score_for_cell 
                # Normalize if scores are too large, or ensure mem_score is scaled appropriately
    return score_map * (grid==-1)


MODULE_FUNCS_VEC: Dict[str, Callable[..., np.ndarray]] = {
    **{f"A{idx}": fn for idx, fn in zip([2,5,6,8],
        [a2_center_radial_vec, a5_adj_density_vec, a6_fixed_position_vec, a8_symmetry_vec])},
    **{f"M{idx}": fn for idx, fn in zip(range(1,12),
        [m1_uni_gap_vec, m2_seq_pattern_vec, m3_diff_band_vec, m4_biaxial_stat_vec,
         m5_bar_focus_vec, m6_neighbor_cycle_vec, m7_bisec_zone_vec, m8_repeat_gap_vec,
         m9_double_rule_overlap_vec, m10_seq_order_match_vec, m11_block_match_vec])},
    **{f"F{idx}": fn for idx, fn in zip([2,3],[f2_row_rotate_vec,f3_col_rotate_vec])},
    **{f"R{idx}": fn for idx, fn_list_tuple in zip([2,7], [ (r2_rev_diff_vec,), (r7_odd_even_dist_vec,) ]) for fn in fn_list_tuple}, # Corrected R series
    "D3": d3_pair_freq_vec,
    "H_ARITHMETIC": h_arithmetic_progression_potential,
    "H_MEMORY": h_memory_based_score,
}

# -----------------------------------------------------------------------------
# 5. Combined score function
# -----------------------------------------------------------------------------
def tensor_flow_score_vec_all(grid: np.ndarray, value_domain_min: int, value_domain_max: int) -> np.ndarray:
    total = np.zeros_like(grid, dtype=float)
    # Heuristics that require value_domain
    domain_aware_heuristics = {"H_ARITHMETIC", "H_MEMORY"}

    for name, fn in MODULE_FUNCS_VEC.items():
        w = MODULE_WEIGHTS.get(name, 0.0)
        if w == 0: # Skip if weight is zero
            continue
        
        # Check if fn is one of the registered functions, not strictly necessary if dict is well-formed
        # if fn not in MODULE_FUNCS_VEC.values(): 
        #     logger.warning(f"Function for {name} not found in values. Skipping.")
        #     continue

        if grid.ndim!=2:
            logger.error(f"Grid is not 2D for heuristic {name}. Skipping.")
            continue
            
        try:
            if name in domain_aware_heuristics:
                score_map = fn(grid.copy(), value_domain_min=value_domain_min, value_domain_max=value_domain_max)
            else:
                score_map = fn(grid.copy()) # Older heuristics don't need domain

            if score_map.shape == grid.shape:
                total += score_map * w
            else:
                logger.error(f"Heuristic {name} returned shape {score_map.shape}, expected {grid.shape}")
        except Exception as e:
            logger.error(f"Error in heuristic {name}: {e}", exc_info=True)
    return total * (grid==-1) # Final mask to be sure

# -----------------------------------------------------------------------------
# 6. Pydantic models & CP-SAT solve step (from version2, with memory & log)
# -----------------------------------------------------------------------------
class GridInput(BaseModel):
    grid: List[List[int]] = Field(..., description="Current grid, -1 for empty")
    num_to_place: int = Field(1, gt=0, description="How many cells to fill")
    value_domain_min: int = Field(1, description="Min value")
    value_domain_max: int = Field(20, description="Max value")

    @validator("grid")
    def check_grid(cls, v):
        if not v or not all(isinstance(row, list) for row in v):
            raise ValueError("Grid must be non-empty list of lists")
        if not v[0]: # Ensure first row is not empty if grid itself is not
             raise ValueError("Grid rows cannot be empty if grid is not empty.")
        length = len(v[0])
        if any(len(r)!=length for r in v):
            raise ValueError("Grid must be rectangular")
        return v

    @validator("value_domain_max")
    def check_domain(cls, vmax, values):
        vmin = values.get("value_domain_min") # Pydantic v1: .get("value_domain_min", None)
        if vmin is not None and vmax < vmin:
            raise ValueError("value_domain_max must >= value_domain_min")
        return vmax

class SolveStepResponse(BaseModel):
    new_grid: List[List[int]]
    chosen_cells: List[Tuple[int,int,int]] # (r, c, value)
    solver_log: str
    status: str
    computed_scores_table: Optional[str] = None
    meta_log_event_id: Optional[str] = None

@app.post("/solve_step", response_model=SolveStepResponse)
async def solve_step_endpoint(grid_input: GridInput, background_tasks: BackgroundTasks):
    grid_np_original = np.array(grid_input.grid) # For memory board ID
    grid_np = grid_np_original.copy() # For modifications
    
    H, W = grid_np.shape
    empties_coords = list(zip(*np.where(grid_np==-1)))
    num_to_place_actual = grid_input.num_to_place

    if not empties_coords:
        return SolveStepResponse(
            new_grid=grid_np.tolist(),
            chosen_cells=[],
            solver_log="No empty cells to fill.",
            status="NO_EMPTY_CELLS"
        )
    
    if num_to_place_actual > len(empties_coords):
        logger.warning(f"Requested to place {num_to_place_actual}, but only {len(empties_coords)} empty. Adjusted.")
        num_to_place_actual = len(empties_coords)
    
    if num_to_place_actual == 0:
        return SolveStepResponse(
            new_grid=grid_np.tolist(),
            chosen_cells=[],
            solver_log="Number of cells to place is 0 (either requested or no empty cells). No action taken.",
            status="NO_ACTION_REQUESTED"
        )

    # Compute heuristic scores
    score_map = await run_in_threadpool(
        tensor_flow_score_vec_all, 
        grid_np.copy(), 
        grid_input.value_domain_min, 
        grid_input.value_domain_max
    )
    
    raw_scores_for_empty_cells = [score_map[r,c] for r,c in empties_coords]
    
    # Scaling for CP-SAT objective
    # Handle case where all raw_scores_for_empty_cells might be 0 or list is empty
    min_raw_score = min(raw_scores_for_empty_cells) if raw_scores_for_empty_cells else 0
    max_raw_score = max(raw_scores_for_empty_cells) if raw_scores_for_empty_cells else 0
    
    # Avoid issues if all scores are identical (e.g., all zero) leading to min_scaled == max_scaled for IntVars
    # We need a range for AddElement's target variable if scores are not all identical.
    # If all scores are same, NewIntVar domain can be just that score.
    
    scaling_factor = 1000.0
    scaled_scores_for_empty_cells = [int(s * scaling_factor) for s in raw_scores_for_empty_cells]
    
    min_scaled_score = min(scaled_scores_for_empty_cells) if scaled_scores_for_empty_cells else 0
    max_scaled_score = max(scaled_scores_for_empty_cells) if scaled_scores_for_empty_cells else 0
    if min_scaled_score == max_scaled_score: # Ensure a valid range for NewIntVar if all scores are the same
        max_scaled_score = min_scaled_score + 1 if not scaled_scores_for_empty_cells or len(scaled_scores_for_empty_cells) == 1 else max_scaled_score


    # Prepare table for logging scores
    table_data = [[r,c, raw_scores_for_empty_cells[i], scaled_scores_for_empty_cells[i]] for i,(r,c) in enumerate(empties_coords)]
    scores_table_str = format_data_as_table(
        table_data,
        headers_option=["Row","Col","Raw Score","Scaled Score"],
        tablefmt="pipe"
    )

    # Build CP-SAT model
    model = cp_model.CpModel()
    # Variables for chosen empty cell indices
    chosen_cell_indices_vars = [model.NewIntVar(0, len(empties_coords)-1, f"idx_{i}") for i in range(num_to_place_actual)]
    # Variables for values to be placed in chosen cells
    assigned_values_vars = [model.NewIntVar(grid_input.value_domain_min, grid_input.value_domain_max, f"val_{i}") for i in range(num_to_place_actual)]

    if num_to_place_actual > 1: # AddAllDifferent needs at least 2 vars, but good practice for >0.
        model.AddAllDifferent(chosen_cell_indices_vars)
        model.AddAllDifferent(assigned_values_vars)
    elif num_to_place_actual == 0: # Should be caught earlier
        pass


    # Objective: Maximize sum of scores of chosen cells
    objective_terms = []
    if num_to_place_actual > 0 and scaled_scores_for_empty_cells: # Ensure list is not empty for AddElement
        for i in range(num_to_place_actual):
            term_var_domain_min = min_scaled_score
            term_var_domain_max = max_scaled_score
            # If list has only one element, AddElement still needs a valid list.
            # scaled_scores_for_empty_cells must be non-empty here.
            
            term = model.NewIntVar(term_var_domain_min, term_var_domain_max, f"term_{i}")
            model.AddElement(chosen_cell_indices_vars[i], scaled_scores_for_empty_cells, term)
            objective_terms.append(term)
    
    if objective_terms:
        model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    # solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    new_grid_filled = grid_np.copy() # Start with current grid for filling
    chosen_actions_list = []
    
    solver_log_message = f"Solver Status: {solver.StatusName(status)}\n"
    if objective_terms:
        obj_val = solver.ObjectiveValue()
        solver_log_message += f"Objective Value (scaled): {obj_val}, Raw Sum Approx: {obj_val/scaling_factor}\n"
    else:
         solver_log_message += "No objective terms to maximize (e.g. num_to_place was 0 or no empty cells).\n"
    solver_log_message += f"Wall Time: {solver.WallTime()}s\n"
    
    meta_log_event: Dict[str, Any] = { # Prepare event for logging
        "request_grid_id": _make_board_id(grid_np_original), # ID of grid *before* this step's changes
        "num_to_place_requested": grid_input.num_to_place,
        "num_to_place_actual": num_to_place_actual,
        "value_domain": [grid_input.value_domain_min, grid_input.value_domain_max],
        "solver_status": solver.StatusName(status),
        "chosen_actions": [], # Will be filled if solution found
        "all_empty_cell_scores_scaled": scaled_scores_for_empty_cells, # For analysis
        "weights_snapshot": MODULE_WEIGHTS.copy()
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) and num_to_place_actual > 0:
        for i in range(num_to_place_actual):
            empty_cell_idx = solver.Value(chosen_cell_indices_vars[i])
            r_chosen, c_chosen = empties_coords[empty_cell_idx]
            value_assigned = solver.Value(assigned_values_vars[i])
            
            new_grid_filled[r_chosen, c_chosen] = value_assigned
            action_detail = (int(r_chosen), int(c_chosen), int(value_assigned))
            chosen_actions_list.append(action_detail)
            
            # Update memory based on the original grid state that led to this decision
            # The "success" of this move is not yet known, assume neutral or use feedback later
            # Score used for memory is the raw heuristic score for the chosen cell
            heuristic_score_for_action = raw_scores_for_empty_cells[empty_cell_idx]
            update_memory(grid_np_original, r_chosen, c_chosen, value_assigned, heuristic_score_for_action, success=True) # Assume success for now

            solver_log_message += (
                f"  Decision {i+1}: Placed {value_assigned} at ({r_chosen},{c_chosen}). "
                f"(EmptyCellIdx: {empty_cell_idx}, RawScore: {heuristic_score_for_action:.3f})\n"
            )
        meta_log_event["chosen_actions"] = chosen_actions_list
        background_tasks.add_task(_save_memory) # Save memory after updates
    else:
        solver_log_message += "No solution found or problem was infeasible/aborted.\n"
        
    solver_log_message += "\nScores for all considered empty cells (before solving):\n" + scores_table_str
    
    # Log the entire event
    meta_logger.log_event(meta_log_event)
    background_tasks.add_task(meta_logger.flush) # Flush logs

    return SolveStepResponse(
        new_grid=new_grid_filled.tolist(),
        chosen_cells=chosen_actions_list,
        solver_log=solver_log_message,
        status=solver.StatusName(status),
        computed_scores_table=scores_table_str,
        meta_log_event_id=meta_log_event.get("log_id")
    )

@app.post("/analyze_scores")
async def analyze_scores_endpoint(grid_input: GridInput):
    grid_np = np.array(grid_input.grid)
    score_map = await run_in_threadpool(
        tensor_flow_score_vec_all, 
        grid_np.copy(),
        grid_input.value_domain_min,
        grid_input.value_domain_max
    )
    empties = list(zip(*np.where(grid_np==-1)))
    data = [[r,c, score_map[r,c]] for r,c in empties]
    table = format_data_as_table(data, headers_option=["Row","Col","Score"], tablefmt="pipe")
    return {
        "message": "Scores computed",
        "scores_table": table,
        "raw_score_map": score_map.tolist()
    }

class FeedbackRequest(BaseModel):
    meta_log_event_id: str = Field(..., description="The ID of the log event this feedback refers to.")
    # Feedback could be about specific chosen_actions within the event, or the overall event
    # For simplicity, let's assume feedback is about the overall utility of the choices in the event
    is_correct_overall: bool = Field(..., description="Were the choices made in this event generally good/correct?")
    # Example: if it's a game, did this move lead to a win or better state?
    # This could be a list of booleans, one for each action in meta_log_event["chosen_actions"]
    # For now, one overall feedback.
    custom_notes: Optional[str] = None

@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest, background_tasks: BackgroundTasks):
    # Conceptual: This feedback could be used to adjust MODULE_WEIGHTS or update memory success rates
    # For now, we just log the feedback.
    feedback_event_data = {
        "feedback_for_event_id": req.meta_log_event_id,
        "is_correct_overall": req.is_correct_overall,
        "custom_notes": req.custom_notes,
        "feedback_type": "user_feedback"
    }
    meta_logger.log_event(feedback_event_data)
    
    # Potentially, use this feedback to update success counts in memory for related past decisions
    # This would require finding the original log event, its board_id, and chosen_actions
    # and then calling update_memory with success=req.is_correct_overall.
    # This is complex and requires careful state management or reading back logs.
    # For now, just log it.

    background_tasks.add_task(meta_logger.flush)
    return {"status": "feedback_recorded", "meta_log_event_id": req.meta_log_event_id}

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down: saving memory, logs, weights")
    _save_memory()
    meta_logger.flush()
    _save_module_weights() # Save weights in case they were modified (though not in this version)

if __name__ == "__main__":
    import uvicorn
    logger.info("Running local server example. Access API at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

    # Example local test (run this script directly if uvicorn is not used for testing)
    # async def main_test():
    #     logger.info("Starting local test example...")
    #     example_grid_data = {
    #         "grid": [
    #             [-1, 12, -1,  8],
    #             [ 5, -1, 19, -1],
    #             [-1,  3, -1, 16],
    #             [10, -1,  7, -1]
    #         ],
    #         "num_to_place": 1,
    #         "value_domain_min": 1,
    #         "value_domain_max": 20
    #     }
        
    #     test_grid_input = GridInput(**example_grid_data)
        
    #     logger.info("\n--- Testing Score Analysis ---")
    #     analysis_response = await analyze_scores_endpoint(test_grid_input)
    #     logger.info(analysis_response["scores_table"])

    #     logger.info("\n--- Testing Solve Step (CP-SAT) ---")
    #     class MockBackgroundTasks:
    #         def add_task(self, func, *args, **kwargs):
    #             logger.info(f"Mock task added: {func.__name__}")
    #             # func(*args, **kwargs) # Optionally run immediately for test

    #     solve_response = await solve_step_endpoint(test_grid_input, MockBackgroundTasks())
    #     logger.info(f"Solver Status: {solve_response.status}")
    #     logger.info(f"Chosen Cells: {solve_response.chosen_cells}")
    #     logger.info("New Grid:\n" + format_data_as_table(solve_response.new_grid, tablefmt="grid"))
    #     logger.info("Solver Log snippet:\n" + "\n".join(solve_response.solver_log.splitlines()[:5]))


    # if __name__ == "__main__" and not os.getenv("FASTAPI_RUN"): # Avoid running test if started by uvicorn
    #      import asyncio
    #      # asyncio.run(main_test())
    #      pass # uvicorn starts the server directly now.

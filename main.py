# ------------------- dependencies -------------------
# pip install fastapi uvicorn ortools tabulate numpy scipy

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
from scipy.signal import convolve2d # For L1 heatmap diffusion

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

# --- Global Configuration for Fair Mode & Weights ---
DEFAULT_MIN_WEIGHT_FLOOR = 0.1 # 最低權重地板值

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
    title="MetaCognitive Scratch Card Solver (v1.3 - Fair Scoring)",
    version="1.3"
)

# -----------------------------------------------------------------------------
# 1. Memory module
# -----------------------------------------------------------------------------
_memory: Dict[str, Dict[str, Any]] = {}

def _make_board_id(grid: np.ndarray) -> str:
    H, W = grid.shape
    empty_count = int(np.sum(grid == -1))
    filled_part_tuple = tuple(map(tuple, grid.tolist()))
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
    bid = _make_board_id(grid)
    key = f"{r}_{c}_{v}"
    if bid not in _memory:
        _memory[bid] = {}
    entry = _memory[bid].setdefault(key, {"count": 0, "total_score": 0.0, "success_count": 0})
    entry["count"] += 1
    entry["total_score"] += score
    if success:
        entry["success_count"] += 1

def mem_score(grid_id: str, r: int, c: int, v: int) -> Tuple[float, int]:
    key = f"{r}_{c}_{v}"
    if grid_id in _memory and key in _memory[grid_id]:
        entry = _memory[grid_id][key]
        count = entry.get("count", 0)
        if count > 0:
            success_rate = entry.get("success_count", 0) / count
            avg_heuristic_score = entry["total_score"] / count
            return avg_heuristic_score * success_rate, count
    return 0.0, 0

_load_memory()

# -----------------------------------------------------------------------------
# 2. Meta-cognition log
# -----------------------------------------------------------------------------
class MetaCognitionLog:
    def __init__(self, path: str):
        self.path = path
        self.buffer: List[Dict[str, Any]] = []

    def log_event(self, event: Dict[str, Any]):
        event["log_id"] = str(uuid.uuid4())
        event["timestamp"] = time.time()
        for k, v_val in list(event.items()): 
            if isinstance(v_val, np.integer): event[k] = int(v_val)
            elif isinstance(v_val, np.floating): event[k] = float(v_val)
            elif isinstance(v_val, np.ndarray): event[k] = v_val.tolist()
            elif isinstance(v_val, tuple): event[k] = list(v_val)
        self.buffer.append(event)

    def flush(self):
        if not self.buffer: return
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
# 3. Module weights management
# -----------------------------------------------------------------------------
MODULE_WEIGHTS: Dict[str, float] = {} 

def _load_module_weights() -> None:
    global MODULE_WEIGHTS
    defaults = {
        "A2": 0.7, "A5": 0.8, "A6": 0.6, "A8": 0.5, "M1": 0.6, "M2": 0.8, "M3": 0.9,
        "M4": 0.5, "M5": 0.5, "M6": 0.5, "M7": 0.5, "M8": 0.5, "M9": 0.6, "M10": 0.5,
        "M11": 0.5, "F2": 0.5, "F3": 0.5, "R2": 0.5, "R7": 0.5, "D3": 0.7,
        "H_ARITHMETIC": 0.8, "H_MEMORY": 1.0,
        "F5": 0.5, "F6": 0.5, "F7": 0.4, "F8": 0.4, "R5": 0.6, "R8": 0.7,
        "P1": 0.7, "P2": 0.6, "P4": 0.5, "L1": 0.6, "L3": 0.5,
    }
    if os.path.exists(MODULE_WEIGHTS_PATH):
        try:
            with open(MODULE_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f) 
            MODULE_WEIGHTS = {**defaults, **loaded}
            for key, value in defaults.items():
                MODULE_WEIGHTS.setdefault(key, value)
            logger.info(f"Loaded module weights from {MODULE_WEIGHTS_PATH}")
        except Exception as e:
            logger.error(f"Error loading weights: {e}. Using default weights.", exc_info=True)
            MODULE_WEIGHTS = defaults.copy()
    else:
        MODULE_WEIGHTS = defaults.copy()
        logger.info(f"Module weights file not found at {MODULE_WEIGHTS_PATH}. Using default weights and creating file.")
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
# 4. Heuristic functions 
# -----------------------------------------------------------------------------
# --- Existing Heuristics (A/M/D series and original F/R) ---
def a2_center_radial_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """A2 中心徑向向量: 評估儲存格到中心點的距離，越近分數越高。"""
    H, W = grid.shape
    if H == 0 or W == 0: return np.zeros_like(grid, dtype=float) * (grid == -1)
    center = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
    rows, cols = np.ogrid[:H, :W]
    dist_sq = (rows - center[0])**2 + (cols - center[1])**2
    dist = np.sqrt(dist_sq)
    max_dist = np.sqrt(((H - 1) / 2.0)**2 + ((W - 1) / 2.0)**2) 
    norm = max_dist if max_dist > 0 else 1.0
    score = 1.0 - (dist / norm)
    return score * (grid == -1)

def a5_adj_density_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """A5 相鄰密度向量: 評估儲存格周圍已填數字的密度。"""
    if grid.size == 0: return np.zeros_like(grid, dtype=float) * (grid == -1)
    padded = np.pad(grid != -1, ((1, 1), (1, 1)), 'constant', constant_values=0)
    dens = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] +
        padded[1:-1, :-2] + padded[1:-1, 2:] 
    ) / 4.0 
    return dens * (grid == -1)

def a6_fixed_position_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """A6 固定位置向量: 給予所有空格一個基礎分數。"""
    return (grid == -1).astype(float)

def a8_symmetry_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """A8 對稱向量: 評估盤面與其水平翻轉後的對稱性。空位相對應空位得分。"""
    if grid.size == 0: return np.zeros_like(grid, dtype=float) * (grid == -1)
    flip = np.fliplr(grid)
    return ((grid == flip).astype(float)) * (grid == -1)

def m1_uni_gap_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M1 單一間隔向量: 評估每行已填數字間隔的均勻度。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if W == 0: return score * (grid == -1)
    for i in range(H):
        filled_indices = np.where(grid[i] != -1)[0]
        if len(filled_indices) > 1:
            gaps = np.diff(filled_indices)
            if len(gaps) > 0:
                mean_gap = np.mean(gaps)
                current_score = 1.0 - np.std(gaps) / (mean_gap if mean_gap > 0 else W)
                score[i, :] = max(0.0, current_score)
            else: 
                score[i, :] = 1.0
    return score * (grid == -1)

def m2_seq_pattern_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M2 序列模式向量: 評估行和列中已填數字序列是否接近等差。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        row_values = np.sort(grid[i][grid[i] != -1])
        if len(row_values) > 2: 
            diffs = np.diff(row_values)
            if len(diffs) > 1: score[i, :] += 1.0 / (1.0 + np.std(diffs))
            elif len(diffs) == 1: score[i, :] += 1.0 
    for j in range(W):
        col_values = np.sort(grid[:, j][grid[:, j] != -1])
        if len(col_values) > 2:
            diffs = np.diff(col_values)
            if len(diffs) > 1: score[:, j] += 1.0 / (1.0 + np.std(diffs))
            elif len(diffs) == 1: score[:, j] += 1.0
    return score * (grid == -1)

def m3_diff_band_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M3 差值區間向量: 評估空格周圍鄰居數字絕對值的平均是否在特定範圍 [5, 20]。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == -1:
                neighbor_abs_vals = []
                for dr_dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr_dc[0], c + dr_dc[1]
                    if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != -1:
                        neighbor_abs_vals.append(abs(grid[nr, nc]))
                if neighbor_abs_vals:
                    mean_abs_val = np.mean(neighbor_abs_vals)
                    score_map[r, c] = 1.0 if 5 <= mean_abs_val <= 20 else 0.3
    return score_map

def m4_biaxial_stat_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M4 雙軸統計向量: 評估空格所在行列的填充密度是否在特定理想範圍 (0.5-0.8)。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    if H == 0 or W == 0: return score_map * (grid == -1)

    row_densities = np.sum(grid != -1, axis=1, dtype=float) / W
    col_densities = np.sum(grid != -1, axis=0, dtype=float) / H
    
    empty_mask = (grid == -1)
    for r in range(H):
        if r < len(row_densities):
            is_row_ideal = (0.5 < row_densities[r] < 0.8)
            for c in range(W):
                if c < len(col_densities): 
                    if empty_mask[r, c] and is_row_ideal and (0.5 < col_densities[c] < 0.8):
                        score_map[r, c] = 1.0
    return score_map

def m5_bar_focus_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M5 長條聚焦向量: 如果某行或列已填數字超過一半，則該行或列的空格得分。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if W > 0:
        for r in range(H):
            if np.sum(grid[r, :] != -1) > W // 2:
                score[r, :] += 1.0
    if H > 0:
        for c in range(W):
            if np.sum(grid[:, c] != -1) > H // 2:
                score[:, c] += 1.0
    return score * (grid == -1)

def m6_neighbor_cycle_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M6 相鄰循環向量: 計算空格的已填鄰居比例。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == -1:
                filled_neighbors_count = 0
                total_valid_neighbors = 0
                for dr_dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr_dc[0], c + dr_dc[1]
                    if 0 <= nr < H and 0 <= nc < W:
                        total_valid_neighbors += 1
                        if grid[nr, nc] != -1:
                            filled_neighbors_count += 1
                if total_valid_neighbors > 0:
                    score_map[r, c] = filled_neighbors_count / total_valid_neighbors
    return score_map

def m7_bisec_zone_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M7 二分區域向量: 評估行列填充數量的標準差，標準差小（分佈均勻）則分數高。"""
    H, W = grid.shape
    score_val = 0.5 
    if H > 0 and W > 0: 
        row_fill_counts = np.sum(grid != -1, axis=1)
        col_fill_counts = np.sum(grid != -1, axis=0)
        denominator = H + W
        if denominator == 0: denominator = 1
        
        std_rows = np.std(row_fill_counts) if len(row_fill_counts) > 0 else 0
        std_cols = np.std(col_fill_counts) if len(col_fill_counts) > 0 else 0

        combined_std_metric = (std_rows + std_cols) / denominator
        score_val = 1.0 / (1.0 + combined_std_metric) 
    return np.full_like(grid, score_val, dtype=float) * (grid == -1)

def m8_repeat_gap_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M8 重複間隔向量: 評估每行已填數字間隔的重複性（標準差相對於平均間隔）。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if W == 0: return score * (grid == -1)
    for i in range(H):
        filled_indices = np.where(grid[i] != -1)[0]
        if len(filled_indices) > 2: 
            gaps = np.diff(filled_indices)
            if len(gaps) > 0:
                mean_gaps = np.mean(gaps)
                current_score = 1.0 - np.std(gaps) / (mean_gaps if mean_gaps > 0 else W)
                score[i, :] = max(0.0, current_score)
            else: 
                score[i,:] = 1.0 if len(filled_indices)>1 else 0.0
        elif len(filled_indices) == 2: 
             score[i,:] = 1.0
    return score * (grid == -1)

def m9_double_rule_overlap_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M9 雙規則重疊向量: 評估空格是否有潛力與左右鄰居形成連接。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == -1:
                if c > 0 and grid[r, c - 1] != -1: score_map[r, c] += 0.5
                if c < W - 1 and grid[r, c + 1] != -1: score_map[r, c] += 0.5
    return score_map

def m10_seq_order_match_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M10 序列順序匹配向量: 如果某行已填數字為嚴格遞增或遞減，則該行空格得分。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        row_values = grid[i][grid[i] != -1]
        if len(row_values) > 1:
            diffs = np.diff(row_values)
            if diffs.size > 0 and (np.all(diffs > 0) or np.all(diffs < 0)):
                score[i, :] = 1.0
    return score * (grid == -1)

def m11_block_match_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M11 區塊匹配向量: 將盤面分塊，空格分數基於其所在區塊的填充密度。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    if H == 0 or W == 0: return score_map * (grid == -1)

    block_size_h = max(1, min(H, H // 4 if H >= 8 else 2))
    block_size_w = max(1, min(W, W // 4 if W >= 8 else 2))

    for r_start in range(0, H, block_size_h):
        for c_start in range(0, W, block_size_w):
            r_end = min(r_start + block_size_h, H)
            c_end = min(c_start + block_size_w, W)
            block = grid[r_start:r_end, c_start:c_end]
            
            if block.size > 0:
                density = np.sum(block != -1) / block.size
                for r_abs in range(r_start, r_end):
                    for c_abs in range(c_start, c_end):
                        if grid[r_abs, c_abs] == -1:
                            score_map[r_abs, c_abs] = max(score_map[r_abs, c_abs], density)
    return score_map * (grid == -1)


def f2_row_rotate_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F2 行旋轉向量: 評估上一行末尾與本行開頭的連接潛力。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if W > 0: 
        for r in range(1, H): 
            if grid[r - 1, W - 1] != -1 and grid[r, 0] == -1:
                score[r, 0] = 1.0
    return score * (grid == -1)

def f3_col_rotate_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F3 列旋轉向量: 評估上一列末尾與本列開頭的連接潛力。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if H > 0: 
        for c in range(1, W): 
            if grid[H - 1, c - 1] != -1 and grid[0, c] == -1:
                score[0, c] = 1.0
    return score * (grid == -1)

def r2_rev_diff_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R2 反向差分向量: 如果某行已填數字為嚴格遞減，則該行空格得分。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        row_values = grid[i][grid[i] != -1]
        if len(row_values) > 1:
            diffs = np.diff(row_values)
            if diffs.size > 0 and np.all(diffs < 0): 
                score[i, :] = 1.0
    return score * (grid == -1)

def r7_odd_even_dist_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R7 奇偶分佈向量: 評估盤面上奇偶數分佈的均衡性。"""
    filled_values = grid[grid != -1]
    score_val = 0.5 
    if filled_values.size > 0:
        try:
            # Ensure only numbers are considered for odd/even counts
            # Attempt to convert to int, skip if not possible (though ideally grid contains only int or -1)
            numeric_values = []
            for x in filled_values:
                try:
                    numeric_values.append(int(x))
                except ValueError:
                    pass # Skip non-integer values
            numeric_values = np.array(numeric_values)

            if numeric_values.size > 0:
                num_odd = np.sum(numeric_values % 2 != 0)
                num_even = np.sum(numeric_values % 2 == 0)
                total_numeric_count = num_odd + num_even
                if total_numeric_count > 0:
                    imbalance_ratio = abs(num_odd - num_even) / total_numeric_count
                    score_val = 1.0 - imbalance_ratio 
                    if imbalance_ratio >= 0.3: score_val = 0.1 
        except Exception: 
            logger.warning("r7_odd_even_dist_vec encountered issues with non-integer data, using default score.")
    return np.full_like(grid, score_val, dtype=float) * (grid == -1)

def d3_pair_freq_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """D3 對頻率向量: 評估空格與其左右鄰居（絕對值為1,9,10）形成連接的潛力。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == -1:
                if c < W - 1 and grid[r, c + 1] != -1 and abs(grid[r, c + 1]) in [1, 9, 10]:
                    score_map[r, c] += 0.5
                if c > 0 and grid[r, c - 1] != -1 and abs(grid[r, c - 1]) in [1, 9, 10]:
                    score_map[r, c] += 0.5
    return score_map

# --- Value-Aware Heuristics ---
def h_arithmetic_progression_potential(grid: np.ndarray, value_domain_min: int, value_domain_max: int, **kwargs) -> np.ndarray:
    """H_ARITHMETIC 等差數列潛力: 評估在空格填入數字後，形成等差數列的最大潛力。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    if not (value_domain_max >= value_domain_min) : return score_map * (grid == -1)

    for r_idx in range(H):
        for c_idx in range(W):
            if grid[r_idx, c_idx] == -1:
                max_cell_progression_score = 0.0
                for candidate_val in range(value_domain_min, value_domain_max + 1):
                    current_candidate_score = 0.0
                    # Horizontal checks
                    if c_idx > 0 and c_idx < W - 1 and grid[r_idx, c_idx-1] != -1 and grid[r_idx, c_idx+1] != -1:
                        if candidate_val - grid[r_idx, c_idx-1] == grid[r_idx, c_idx+1] - candidate_val:
                            current_candidate_score += 1.0
                    if c_idx < W - 2 and grid[r_idx, c_idx+1] != -1 and grid[r_idx, c_idx+2] != -1:
                        if grid[r_idx, c_idx+1] - candidate_val == grid[r_idx, c_idx+2] - grid[r_idx, c_idx+1]:
                            current_candidate_score += 0.75
                    if c_idx > 1 and grid[r_idx, c_idx-1] != -1 and grid[r_idx, c_idx-2] != -1:
                        if grid[r_idx, c_idx-1] - grid[r_idx, c_idx-2] == candidate_val - grid[r_idx, c_idx-1]:
                            current_candidate_score += 0.75
                    # Vertical checks
                    if r_idx > 0 and r_idx < H - 1 and grid[r_idx-1, c_idx] != -1 and grid[r_idx+1, c_idx] != -1:
                        if candidate_val - grid[r_idx-1, c_idx] == grid[r_idx+1, c_idx] - candidate_val:
                            current_candidate_score += 1.0
                    if r_idx < H - 2 and grid[r_idx+1, c_idx] != -1 and grid[r_idx+2, c_idx] != -1:
                        if grid[r_idx+1, c_idx] - candidate_val == grid[r_idx+2, c_idx] - grid[r_idx+1, c_idx]:
                            current_candidate_score += 0.75
                    if r_idx > 1 and grid[r_idx-1, c_idx] != -1 and grid[r_idx-2, c_idx] != -1:
                        if grid[r_idx-1, c_idx] - grid[r_idx-2, c_idx] == candidate_val - grid[r_idx-1, c_idx]:
                             current_candidate_score += 0.75
                    
                    if current_candidate_score > max_cell_progression_score:
                        max_cell_progression_score = current_candidate_score
                score_map[r_idx, c_idx] = max_cell_progression_score
    return score_map * (grid == -1)

def h_memory_based_score(grid: np.ndarray, value_domain_min: int, value_domain_max: int, **kwargs) -> np.ndarray:
    """H_MEMORY 記憶啟發分數: 利用歷史記憶評估在空格填入不同值的最大成功調整後平均分。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    current_grid_id = _make_board_id(grid)
    if not (value_domain_max >= value_domain_min) : return score_map * (grid == -1)

    for r_idx in range(H):
        for c_idx in range(W):
            if grid[r_idx, c_idx] == -1:
                max_mem_score_for_cell = 0.0
                for val_candidate in range(value_domain_min, value_domain_max + 1):
                    avg_score, count = mem_score(current_grid_id, r_idx, c_idx, val_candidate)
                    if count > 0 and avg_score > max_mem_score_for_cell:
                        max_mem_score_for_cell = avg_score
                score_map[r_idx, c_idx] = max_mem_score_for_cell
    return score_map * (grid == -1)

# ─────────────────────────────────────────────────────────────────────────────
# ── 新增活逻辑模块 (New Live Logic Modules - F, R, P, L series) ───────────────
# ─────────────────────────────────────────────────────────────────────────────

# --- F Series (統計與變異分析 - Statistics and Variance Analysis) ---
def f5_row_density_stats_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F5 行密度統計: 計算每行已填充儲存格的密度，並將此密度賦予該行所有空格。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    if W == 0: return score_map * (grid == -1) # Avoid division by zero if no columns
    
    row_densities = np.sum(grid != -1, axis=1, dtype=float) / W
    for r in range(H):
        score_map[r, :] = row_densities[r] # Apply row density to all cells in that row
    return score_map * (grid == -1) # Mask to apply only to empty cells

def f6_col_density_stats_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F6 列密度統計: 計算每列已填充儲存格的密度，並將此密度賦予該列所有空格。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    if H == 0: return score_map * (grid == -1) # Avoid division by zero if no rows

    col_densities = np.sum(grid != -1, axis=0, dtype=float) / H
    for c in range(W):
        score_map[:, c] = col_densities[c] # Apply column density to all cells in that column
    return score_map * (grid == -1) # Mask to apply only to empty cells

def f7_horizontal_value_variance_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F7 橫向號碼變異: 計算每行中已填充數字的變異數(方差)。變異越小，該行空格得分越高。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    for r in range(H):
        row_values = grid[r, grid[r, :] != -1] # Get filled values in the current row
        if len(row_values) > 1: # Variance requires at least 2 values
            variance = np.var(row_values)
            # Score is inversely proportional to variance; add 1 to avoid division by zero for variance=0
            score_map[r, :] = 1.0 / (1.0 + variance) 
        elif len(row_values) == 1: # Only one value, no variance, assign high score
            score_map[r, :] = 1.0
        # If len(row_values) == 0, score remains 0 for that row
    return score_map * (grid == -1)

def f8_vertical_value_variance_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F8 直向號碼變異: 計算每列中已填充數字的變異數(方差)。變異越小，該列空格得分越高。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    for c in range(W):
        col_values = grid[grid[:, c] != -1, c] # Get filled values in the current column
        if len(col_values) > 1:
            variance = np.var(col_values)
            score_map[:, c] = 1.0 / (1.0 + variance)
        elif len(col_values) == 1:
            score_map[:, c] = 1.0
    return score_map * (grid == -1)

# --- R Series (出現與加權排序 - Appearance and Weighted Sorting) ---
def r5_appearance_order_stats_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R5 出現次序統計: 簡化實現：賦予較低行/列索引的空格更高的基礎分數，模擬對早期位置的偏好。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    if H == 0 or W == 0: return score_map * (grid == -1)

    # Scores decrease from top-left (1.0) to bottom-right (approaching 0).
    row_component = (1.0 - np.arange(H, dtype=float) / H)[:, np.newaxis] 
    col_component = (1.0 - np.arange(W, dtype=float) / W)[np.newaxis, :]
    
    # Combine row and column positional bias (e.g., average)
    # Ensure broadcasting works: tile row_component to (H,W), col_component to (H,W)
    base_score_map = (np.tile(row_component, (1, W)) + np.tile(col_component, (H, 1))) / 2.0
    score_map = base_score_map
    return score_map * (grid == -1)

def r8_frequency_weighted_integration_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R8 頻次加權整合: 空格的分數基於其周圍8個鄰居中罕見數字的平均罕見度。罕見度 = 1 - 標準化頻次。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    
    filled_values = grid[grid != -1]
    if filled_values.size == 0: # If no numbers are filled, all cells get 0 score from this heuristic
        return score_map * (grid == -1)
        
    value_counts = Counter(filled_values)
    total_occurrences = filled_values.size # Sum of all counts
    
    # Rarity: 1 for numbers that appear once (if total_occurrences > 1), less for more frequent.
    # If only one unique number that fills the whole board, its rarity would be 0.
    value_rarity = {}
    if total_occurrences > 0:
        for val, count in value_counts.items():
            # Normalize frequency: count / total_occurrences
            # Rarity: 1 - normalized_frequency
            # If a number is unique and total_occurrences > 1, its count is 1.
            # Rarity = 1 - (1/total_occurrences). If it's the only number, rarity = 0.
            if total_occurrences == count and len(value_counts)==1 : # only one distinct number filling everything
                value_rarity[val] = 0.0
            else:
                value_rarity[val] = 1.0 - (count / total_occurrences)

    for r_idx in range(H):
        for c_idx in range(W):
            if grid[r_idx, c_idx] == -1: # Only for empty cells
                sum_neighbor_rarity = 0.0
                num_counted_neighbors = 0
                for dr_offset in [-1, 0, 1]:
                    for dc_offset in [-1, 0, 1]:
                        if dr_offset == 0 and dc_offset == 0: continue # Skip the cell itself
                        
                        nr, nc = r_idx + dr_offset, c_idx + dc_offset
                        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != -1:
                            neighbor_val = grid[nr, nc]
                            sum_neighbor_rarity += value_rarity.get(neighbor_val, 0.0) # Default 0 rarity if not in map (should not happen)
                            num_counted_neighbors += 1
                if num_counted_neighbors > 0:
                    score_map[r_idx, c_idx] = sum_neighbor_rarity / num_counted_neighbors
    return score_map * (grid == -1)

# --- P Series (記憶比對與向量特徵 - Memory Comparison and Vector Features) ---
def p1_similar_memory_comparison_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """P1 相似記憶比對: 佔位符：基於全局盤面填充密度的分數，傾向於適度填充 (0.25-0.75) 的盤面。"""
    H, W = grid.shape
    score_val = 0.0 # Default score if grid is empty or too small
    if H * W > 0:
        density = np.sum(grid != -1) / (H * W)
        # Score peaks at 0.5 density (score 1.0), linear fall-off to 0 at 0.25 and 0.75 density
        if 0.25 <= density <= 0.75:
            score_val = 1.0 - abs(density - 0.5) / 0.25 
        # else score_val remains 0 if outside this preferred density range
    return np.full_like(grid, score_val, dtype=float) * (grid == -1)

def p2_structural_feature_vector_matching_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """P2 結構特徵向量匹配: 佔位符：計算四象限填充密度，並基於其均衡性（低方差）評分。"""
    H, W = grid.shape
    score_val = 0.5 # Default score for small or un-analyzable grids
    if H >= 2 and W >= 2: # Need at least 2x2 to meaningfully divide into quadrants
        mid_h, mid_w = H // 2, W // 2
        # Define quadrant slices carefully to handle odd/even dimensions
        quadrant_slices = [
            (slice(0, mid_h), slice(0, mid_w)),                             # Top-left
            (slice(0, mid_h), slice(mid_w, W)),                             # Top-right
            (slice(mid_h, H), slice(0, mid_w)),                             # Bottom-left
            (slice(mid_h, H), slice(mid_w, W)),                             # Bottom-right
        ]
        quadrant_densities = []
        for r_slice, c_slice in quadrant_slices:
            quadrant = grid[r_slice, c_slice]
            if quadrant.size > 0: # Ensure quadrant is not empty (e.g. if H or W is 1 after slicing)
                quadrant_densities.append(np.sum(quadrant != -1) / quadrant.size)
        
        if len(quadrant_densities) > 1: # Need more than one density to calculate variance
            density_variance = np.var(quadrant_densities)
            # Score is inversely related to variance; higher variance means less balance, lower score
            # Add a small epsilon to prevent division by zero if variance is very small
            score_val = 1.0 / (1.0 + density_variance * 10.0 + 1e-9) 
        elif len(quadrant_densities) == 1: # Effectively one large quadrant
            score_val = 0.75 # Moderately good score for a single, analyzable quadrant
            
    return np.full_like(grid, score_val, dtype=float) * (grid == -1)

def p4_local_structure_residual_analysis_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """P4 局部結構殘差分析: 分析 2x2 區塊的填充模式。某些預定義模式（如對角線）得分較高。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    if H < 2 or W < 2: return score_map * (grid == -1) # Need at least 2x2 grid

    # Define some 2x2 binary patterns (1=filled, 0=empty) and their "desirability" scores
    # Key: tuple of tuples, e.g., ((row1_col1, row1_col2), (row2_col1, row2_col2))
    target_patterns_scores = {
        ((1,0),(0,1)): 1.0, # Diagonal \ (top-left to bottom-right)
        ((0,1),(1,0)): 1.0, # Diagonal / (top-right to bottom-left)
        ((1,1),(0,0)): 0.7, # Top row filled in 2x2
        ((0,0),(1,1)): 0.7, # Bottom row filled in 2x2
        ((1,0),(1,0)): 0.7, # Left column filled in 2x2
        ((0,1),(0,1)): 0.7, # Right column filled in 2x2
        ((1,1),(1,0)): 0.8, # L-shape variant 1
        ((1,1),(0,1)): 0.8, # L-shape variant 2
        ((1,0),(1,1)): 0.8, # L-shape variant 3
        ((0,1),(1,1)): 0.8, # L-shape variant 4
    }
    default_block_pattern_score = 0.1 # Score for non-matching or less interesting patterns

    for r_start in range(H - 1): # Iterate to form top-left corners of 2x2 blocks
        for c_start in range(W - 1):
            current_block = grid[r_start:r_start+2, c_start:c_start+2]
            # Convert block to a binary pattern (tuple of tuples) for dictionary lookup
            binary_block_pattern = tuple(map(tuple, (current_block != -1).astype(int)))
            
            block_inherent_score = target_patterns_scores.get(binary_block_pattern, default_block_pattern_score)
            
            # Apply this score to any empty cells within this 2x2 block
            # If a cell is part of multiple overlapping blocks, it gets the max score from those blocks
            for dr_offset in range(2):
                for dc_offset in range(2):
                    abs_r, abs_c = r_start + dr_offset, c_start + dc_offset
                    if grid[abs_r, abs_c] == -1: # If the cell in the original grid is empty
                        score_map[abs_r, abs_c] = max(score_map[abs_r, abs_c], block_inherent_score)
                        
    return score_map * (grid == -1) # Ensure final mask, although logic above applies to empty cells

# --- L Series (熱區圖形邏輯 - Heatmap Pattern Logic) ---
def l1_heatmap_diffusion_logic_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """L1 熱區轉換擴散邏輯: 已填充儲存格為熱源(1.0)，熱量透過卷積向周圍空格擴散。"""
    if grid.size == 0: return np.zeros_like(grid, dtype=float) * (grid == -1)
    
    # Kernel for diffusion: e.g., Gaussian blur or simple average-like filter
    # This kernel gives more weight to cardinal neighbors than diagonal ones
    diffusion_kernel = np.array([[0.5, 1.0, 0.5],
                                 [1.0, 0.0, 1.0], # Center is 0 as we diffuse *from* sources
                                 [0.5, 1.0, 0.5]]) 
    kernel_sum = np.sum(diffusion_kernel)
    if kernel_sum > 0 : diffusion_kernel = diffusion_kernel / kernel_sum # Normalize kernel

    # Input for diffusion: filled cells are heat sources (1.0), empty cells are 0.0
    heat_source_map = (grid != -1).astype(float)
    
    # Apply convolution to simulate heat diffusion
    # 'same' mode ensures output map has same dimensions as input
    # 'symm' boundary condition reflects values at boundaries, can be 'fill' with fillvalue=0 too
    diffused_heat_map = convolve2d(heat_source_map, diffusion_kernel, mode='same', boundary='symm')
    
    # Normalize the resulting heatmap scores to [0,1] for consistency
    max_heat_value = np.max(diffused_heat_map)
    if max_heat_value > 0:
        diffused_heat_map /= max_heat_value
        
    return diffused_heat_map * (grid == -1) # Apply scores only to empty cells

def l3_pattern_block_rotation_analysis_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """L3 圖形分塊輪替分析: 佔位符：檢查 2x2 區塊的均一性或簡單的90度旋轉對稱性。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    if H < 2 or W < 2: return score_map * (grid == -1) # Grid too small for 2x2 blocks

    for r_start in range(H - 1):
        for c_start in range(W - 1):
            current_block = grid[r_start:r_start+2, c_start:c_start+2] # Current 2x2 block
            current_block_score = 0.0
            
            # Check for uniformity (all cells in block have same value)
            unique_vals_in_block = np.unique(current_block)
            if len(unique_vals_in_block) == 1:
                # High score if block is uniformly empty or uniformly filled with some number
                current_block_score = 1.0 if unique_vals_in_block[0] == -1 else 0.8 
            else:
                # Check for 90-degree rotational symmetry of the filled/empty pattern
                binary_pattern_block = (current_block != -1).astype(int) # 0 for empty, 1 for filled
                rotated_90_pattern = np.rot90(binary_pattern_block)
                if np.array_equal(binary_pattern_block, rotated_90_pattern):
                    current_block_score = 0.7 # Score for 90-degree rotational symmetry
                # Could add checks for 180-degree symmetry if not 90-deg symmetric:
                # rotated_180_pattern = np.rot90(binary_pattern_block, 2) # Rotate 180 degrees
                # if np.array_equal(binary_pattern_block, rotated_180_pattern):
                #    current_block_score = max(current_block_score, 0.5) # Score for 180-deg symmetry
            
            # Apply this block's score to any empty cells within it
            for dr_offset in range(2):
                for dc_offset in range(2):
                    abs_r, abs_c = r_start + dr_offset, c_start + dc_offset
                    if grid[abs_r, abs_c] == -1: # If the cell in the original grid is empty
                        score_map[abs_r, abs_c] = max(score_map[abs_r, abs_c], current_block_score)
                        
    return score_map * (grid == -1) # Final mask


# --- MODULE_FUNCS_VEC Registration ---
MODULE_FUNCS_VEC: Dict[str, Callable[..., np.ndarray]] = {
    # A Series
    "A2": a2_center_radial_vec, "A5": a5_adj_density_vec,
    "A6": a6_fixed_position_vec, "A8": a8_symmetry_vec,
    # M Series
    "M1": m1_uni_gap_vec, "M2": m2_seq_pattern_vec, "M3": m3_diff_band_vec,
    "M4": m4_biaxial_stat_vec, "M5": m5_bar_focus_vec, "M6": m6_neighbor_cycle_vec,
    "M7": m7_bisec_zone_vec, "M8": m8_repeat_gap_vec, "M9": m9_double_rule_overlap_vec,
    "M10": m10_seq_order_match_vec, "M11": m11_block_match_vec,
    # Original F Series
    "F2": f2_row_rotate_vec, "F3": f3_col_rotate_vec,
    # Original R Series
    "R2": r2_rev_diff_vec, "R7": r7_odd_even_dist_vec,
    # D Series
    "D3": d3_pair_freq_vec,
    # Value-Aware Heuristics
    "H_ARITHMETIC": h_arithmetic_progression_potential,
    "H_MEMORY": h_memory_based_score,
    # New F Series (活逻辑模块)
    "F5": f5_row_density_stats_vec, "F6": f6_col_density_stats_vec,
    "F7": f7_horizontal_value_variance_vec, "F8": f8_vertical_value_variance_vec,
    # New R Series (活逻辑模块)
    "R5": r5_appearance_order_stats_vec, "R8": r8_frequency_weighted_integration_vec,
    # New P Series (活逻辑模块)
    "P1": p1_similar_memory_comparison_vec, "P2": p2_structural_feature_vector_matching_vec,
    "P4": p4_local_structure_residual_analysis_vec,
    # New L Series (活逻辑模块)
    "L1": l1_heatmap_diffusion_logic_vec, "L3": l3_pattern_block_rotation_analysis_vec,
}

# -----------------------------------------------------------------------------
# 5. Combined score function with Normalization and Fair Mode
# -----------------------------------------------------------------------------
def tensor_flow_score_vec_all(
    grid: np.ndarray, 
    value_domain_min: int, 
    value_domain_max: int,
    fair_mode: bool = False, 
    min_weight_floor: float = DEFAULT_MIN_WEIGHT_FLOOR 
) -> np.ndarray:
    """
    計算所有啟發式模組的加權總分。
    在 'fair_mode' 下，分數會進行歸一化，並應用最低權重下限。
    """
    if grid.ndim != 2:
        logger.error("輸入的 grid 必須是二維陣列。返回零分圖。")
        return np.zeros_like(grid, dtype=float) if isinstance(grid, np.ndarray) and grid.ndim == 2 else np.array([[]], dtype=float)
    if grid.size == 0:
        return np.array([[]], dtype=float) if grid.ndim == 2 else np.array([], dtype=float)

    total_score_map = np.zeros(grid.shape, dtype=float)
    domain_aware_heuristic_names = {"H_ARITHMETIC", "H_MEMORY"}
    empty_cell_mask = (grid == -1) # Cache this mask

    # For Req 4 (Dynamic Weight Adjustment):
    # Data for dynamic weight adjustment could be gathered here or from MetaCognitionLog.
    # For example, log each module's raw_score_map and normalized_score_map for analysis against outcomes.
    # module_intermediate_outputs = {} 

    for name, heuristic_func in MODULE_FUNCS_VEC.items():
        configured_weight = MODULE_WEIGHTS.get(name, 0.0)
        
        if configured_weight == 0.0 and not fair_mode: 
            continue # Skip if weight is 0 and not in fair_mode (where floor applies)
            
        try:
            kwargs_for_func = {}
            if name in domain_aware_heuristic_names:
                kwargs_for_func['value_domain_min'] = value_domain_min
                kwargs_for_func['value_domain_max'] = value_domain_max
            
            # It's crucial that heuristic_func does not modify the grid passed to it.
            # Passing grid.copy() ensures this.
            raw_score_map = heuristic_func(grid.copy(), **kwargs_for_func).astype(float)

            if raw_score_map.shape != grid.shape:
                logger.error(f"啟發式模組 {name} 返回的形狀 {raw_score_map.shape} 與預期 {grid.shape} 不符。跳過此模組。")
                continue

            current_contribution: np.ndarray
            
            # Apply mask *before* normalization to ensure normalization is based on relevant scores
            relevant_raw_scores = raw_score_map[empty_cell_mask]

            if fair_mode:
                # --- 1. 分數歸一化 (Min-Max Normalization to [0,1]) ---
                normalized_relevant_scores = np.zeros_like(relevant_raw_scores, dtype=float)
                if relevant_raw_scores.size > 0:
                    min_val = np.min(relevant_raw_scores)
                    max_val = np.max(relevant_raw_scores)
                    if max_val > min_val:
                        normalized_relevant_scores = (relevant_raw_scores - min_val) / (max_val - min_val)
                    elif max_val == min_val: 
                        # If all relevant scores are the same, map them to 0.5 (neutral) or 0 or 1.
                        # Mapping to 0.5 suggests it fired uniformly. If min_val is 0, then map to 0. Else 1.
                        # A simple approach: if all same, they are all "average" or "equally important".
                        normalized_relevant_scores = np.full_like(relevant_raw_scores, 0.5, dtype=float)
                
                processed_score_map_for_empty_cells = normalized_relevant_scores
                
                # --- 2. 最低權重地板 (Minimum Weight Floor) ---
                # Apply floor to the configured weight.
                effective_weight = max(configured_weight, min_weight_floor)
                
                final_contribution_for_empty_cells = processed_score_map_for_empty_cells * effective_weight
            else: # Original mode: raw_score * configured_weight
                # Use relevant_raw_scores (scores of empty cells) directly
                final_contribution_for_empty_cells = relevant_raw_scores * configured_weight
            
            # Place the contributions back into the full map for accumulation
            # Ensure only empty cells get these scores; other cells remain 0 for this heuristic's turn.
            current_heuristic_total_contribution = np.zeros_like(grid, dtype=float)
            current_heuristic_total_contribution[empty_cell_mask] = final_contribution_for_empty_cells
            total_score_map += current_heuristic_total_contribution
            
            # For Req 4 (Dynamic Weight Adjustment) - logging for analysis:
            # module_intermediate_outputs[name] = {
            #     "raw_scores_empty": relevant_raw_scores.tolist(),
            #     "normalized_scores_empty": normalized_relevant_scores.tolist() if fair_mode else None,
            #     "effective_weight": effective_weight if fair_mode else configured_weight,
            #     "final_contribution_empty": final_contribution_for_empty_cells.tolist()
            # }

        except Exception as e:
            logger.error(f"執行或處理啟發式模組 {name} 時出錯: {e}", exc_info=True)
            
    # The total_score_map should inherently only have scores on empty_cell_mask positions
    # due to the way contributions are added. No final global mask needed if logic is correct.
    return total_score_map

# -----------------------------------------------------------------------------
# 6. Pydantic models & CP-SAT solve step
# -----------------------------------------------------------------------------
class GridInput(BaseModel):
    grid: List[List[int]] = Field(..., description="Current grid, -1 for empty")
    num_to_place: int = Field(1, gt=0, description="How many cells to fill")
    value_domain_min: int = Field(1, description="Min value for filling cells")
    value_domain_max: int = Field(20, description="Max value for filling cells")
    fair_mode: bool = Field(False, description="Enable fair mode for heuristic scoring (normalization + weight floor)")
    min_weight_floor_override: Optional[float] = Field(None, ge=0, le=1, description="Override global min_weight_floor if in fair_mode. Default: " + str(DEFAULT_MIN_WEIGHT_FLOOR))

    @validator("grid")
    def check_grid_is_valid(cls, v_grid: List[List[int]]):
        if not v_grid: raise ValueError("Grid cannot be empty list.")
        if not isinstance(v_grid[0], list): raise ValueError("Grid must be a list of lists.")
        first_row_len = len(v_grid[0])
        if len(v_grid) == 1 and first_row_len == 0: return v_grid # Allow [[]]
        if not all(len(row) == first_row_len for row in v_grid):
            raise ValueError("Grid must be rectangular (all rows same length).")
        return v_grid

    @validator("value_domain_max")
    def check_value_domain_max_ge_min(cls, v_max: int, values: Dict[str, Any]):
        v_min = values.get("value_domain_min")
        if v_min is not None and v_max < v_min:
            raise ValueError("value_domain_max must be greater than or equal to value_domain_min.")
        return v_max

class SolveStepResponse(BaseModel):
    new_grid: List[List[int]]
    chosen_cells: List[Tuple[int,int,int]]
    solver_log: str
    status: str
    computed_scores_table: Optional[str] = None
    meta_log_event_id: Optional[str] = None
    active_fair_mode: bool 

@app.post("/solve_step", response_model=SolveStepResponse)
async def solve_step_endpoint(grid_input: GridInput, background_tasks: BackgroundTasks):
    grid_np_original = np.array(grid_input.grid)
    grid_np_current = grid_np_original.copy()
    
    H, W = grid_np_current.shape
    empty_cell_coordinates = list(zip(*np.where(grid_np_current == -1)))
    num_to_place_requested = grid_input.num_to_place
    
    actual_num_to_place = num_to_place_requested
    active_fair_mode_for_response = grid_input.fair_mode # Capture for response

    if not empty_cell_coordinates:
        return SolveStepResponse(
            new_grid=grid_np_current.tolist(), chosen_cells=[], 
            solver_log="No empty cells to fill.", status="NO_EMPTY_CELLS",
            active_fair_mode=active_fair_mode_for_response
        )
    
    if actual_num_to_place > len(empty_cell_coordinates):
        logger.warning(f"Requested {actual_num_to_place} places, but only {len(empty_cell_coordinates)} empty. Adjusted.")
        actual_num_to_place = len(empty_cell_coordinates)
    
    if actual_num_to_place == 0:
        return SolveStepResponse(
            new_grid=grid_np_current.tolist(), chosen_cells=[], 
            solver_log="Num to place is 0. No action.", status="NO_ACTION_REQUESTED",
            active_fair_mode=active_fair_mode_for_response
            )

    current_min_weight_floor = grid_input.min_weight_floor_override if grid_input.min_weight_floor_override is not None else DEFAULT_MIN_WEIGHT_FLOOR

    combined_score_map = await run_in_threadpool(
        tensor_flow_score_vec_all, 
        grid_np_current.copy(), 
        grid_input.value_domain_min, 
        grid_input.value_domain_max,
        grid_input.fair_mode,
        current_min_weight_floor
    )
    
    effective_scores_of_empty_cells = [combined_score_map[r,c] for r,c in empty_cell_coordinates]
    
    scaling_factor = 1000.0 
    scaled_effective_scores = [int(s * scaling_factor) for s in effective_scores_of_empty_cells]
    
    min_s = min(scaled_effective_scores) if scaled_effective_scores else 0
    max_s = max(scaled_effective_scores) if scaled_effective_scores else 0
    if min_s == max_s : max_s = min_s + 1 if scaled_effective_scores else 1 # Ensure range for IntVar, handle empty list

    score_table_for_log = format_data_as_table(
        [[r,c, effective_scores_of_empty_cells[i], scaled_effective_scores[i]] for i,(r,c) in enumerate(empty_cell_coordinates)],
        headers_option=["Row","Col","Effective Score","Scaled Score"], tablefmt="pipe"
    )

    model = cp_model.CpModel()
    chosen_indices_vars = [model.NewIntVar(0, len(empty_cell_coordinates)-1, f"idx_{i}") for i in range(actual_num_to_place)]
    chosen_values_vars = [model.NewIntVar(grid_input.value_domain_min, grid_input.value_domain_max, f"val_{i}") for i in range(actual_num_to_place)]

    if actual_num_to_place > 1:
        model.AddAllDifferent(chosen_indices_vars)
        model.AddAllDifferent(chosen_values_vars)
    
    obj_terms = []
    if actual_num_to_place > 0 and scaled_effective_scores:
        for i in range(actual_num_to_place):
            term = model.NewIntVar(min_s, max_s, f"term_score_{i}")
            model.AddElement(chosen_indices_vars[i], scaled_effective_scores, term)
            obj_terms.append(term)
    
    if obj_terms: model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    final_filled_grid = grid_np_current.copy()
    actions_taken = []
    solver_log_message = f"Solver Status: {solver.StatusName(status)} (Fair Mode: {grid_input.fair_mode})\n"

    if obj_terms and status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        obj_val = solver.ObjectiveValue()
        solver_log_message += f"Objective (scaled): {obj_val}, Effective Raw Sum Approx: {obj_val/scaling_factor if scaling_factor !=0 else 'N/A'}\n"
    elif obj_terms:
        solver_log_message += "Objective Value: N/A (No optimal/feasible solution found)\n"
    else:
         solver_log_message += "No objective terms to maximize.\n"
    solver_log_message += f"Wall Time: {solver.WallTime()}s\n"
    
    current_meta_log_event: Dict[str, Any] = {
        "request_grid_id": _make_board_id(grid_np_original),
        "fair_mode_active": grid_input.fair_mode,
        "min_weight_floor_used": current_min_weight_floor if grid_input.fair_mode else None,
        "num_to_place_requested": grid_input.num_to_place,
        "num_to_place_actual": actual_num_to_place,
        "value_domain": [grid_input.value_domain_min, grid_input.value_domain_max],
        "solver_status": solver.StatusName(status), "chosen_actions": [],
        "all_empty_cell_effective_scores": effective_scores_of_empty_cells, 
        "weights_snapshot": MODULE_WEIGHTS.copy()
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) and actual_num_to_place > 0:
        for i in range(actual_num_to_place):
            selected_empty_idx = solver.Value(chosen_indices_vars[i])
            r_chosen, c_chosen = empty_cell_coordinates[selected_empty_idx]
            value_assigned = solver.Value(chosen_values_vars[i])
            final_filled_grid[r_chosen, c_chosen] = value_assigned
            action_detail = (int(r_chosen), int(c_chosen), int(value_assigned))
            actions_taken.append(action_detail)
            effective_score_for_action = effective_scores_of_empty_cells[selected_empty_idx]
            update_memory(grid_np_original, r_chosen, c_chosen, value_assigned, effective_score_for_action, success=True)
            solver_log_message += (f"  Decision {i+1}: Placed {value_assigned} at ({r_chosen},{c_chosen}). (EffectiveScore: {effective_score_for_action:.3f})\n")
        current_meta_log_event["chosen_actions"] = actions_taken
        background_tasks.add_task(_save_memory)
    else:
        solver_log_message += "No solution or problem infeasible/aborted, or no cells to place.\n"
    solver_log_message += "\nScores for all considered empty cells (before solving):\n" + score_table_for_log
    meta_logger.log_event(current_meta_log_event)
    background_tasks.add_task(meta_logger.flush)

    return SolveStepResponse(
        new_grid=final_filled_grid.tolist(), chosen_cells=actions_taken,
        solver_log=solver_log_message, status=solver.StatusName(status),
        computed_scores_table=score_table_for_log,
        meta_log_event_id=current_meta_log_event.get("log_id"),
        active_fair_mode=active_fair_mode_for_response
    )

@app.post("/analyze_scores")
async def analyze_scores_endpoint(grid_input: GridInput): 
    grid_np = np.array(grid_input.grid)
    if grid_np.size == 0 and not (grid_input.grid and isinstance(grid_input.grid[0],list) and not grid_input.grid[0]):
        return {"message": "Empty grid provided.", "scores_table": "No data to format.", 
                "raw_score_map": [[]], "active_fair_mode": grid_input.fair_mode}

    current_min_weight_floor = grid_input.min_weight_floor_override if grid_input.min_weight_floor_override is not None else DEFAULT_MIN_WEIGHT_FLOOR

    score_map = await run_in_threadpool(
        tensor_flow_score_vec_all, 
        grid_np.copy(),
        grid_input.value_domain_min,
        grid_input.value_domain_max,
        grid_input.fair_mode,
        current_min_weight_floor
    )
    empties = list(zip(*np.where(grid_np == -1)))
    data_for_table = [[r,c, score_map[r,c]] for r,c in empties if grid_np[r,c] == -1] 
    table_str = format_data_as_table(data_for_table, headers_option=["Row","Col","Effective Score"], tablefmt="pipe")
    return {
        "message": "Scores computed", "scores_table": table_str, 
        "raw_score_map": score_map.tolist(), 
        "active_fair_mode": grid_input.fair_mode
    }

class FeedbackRequest(BaseModel):
    meta_log_event_id: str = Field(..., description="The ID of the log event this feedback refers to.")
    is_correct_overall: bool = Field(..., description="Were the choices made in this event generally good/correct?")
    custom_notes: Optional[str] = None

@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest, background_tasks: BackgroundTasks):
    feedback_event_data = {
        "feedback_for_event_id": req.meta_log_event_id,
        "is_correct_overall": req.is_correct_overall,
        "custom_notes": req.custom_notes,
        "feedback_type": "user_feedback"
    }
    meta_logger.log_event(feedback_event_data)
    background_tasks.add_task(meta_logger.flush)
    logger.info(f"Feedback recorded for {req.meta_log_event_id}. Dynamic weight adjustment is a future enhancement based on such feedback.")
    return {"status": "feedback_recorded", "meta_log_event_id": req.meta_log_event_id}

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down: saving memory, logs, weights")
    _save_memory()
    meta_logger.flush()
    _save_module_weights()

if __name__ == "__main__":
    import uvicorn
    logger.info("Running FastAPI server. Access API at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) # Use string for app for reload

    # --- Conceptual Unit Test Descriptions / Usage Scenarios ---
    # (These would typically be in separate test files using pytest or unittest)

    # Test Case 1: Fair mode vs Original mode with extreme weights
    #   Objective: Verify that a module with a very low configured weight but high raw score
    #              can influence the decision in fair_mode due to normalization and weight floor.
    #   Setup:
    #     - Grid: A simple 3x3 grid.
    #     - Heuristic A (e.g., A2): Outputs a high raw score (e.g., 100) for cell_A. Configured weight = 0.01.
    #     - Heuristic B (e.g., M1): Outputs a moderate raw score (e.g., 10) for cell_B. Configured weight = 1.0.
    #     - Other heuristics: Output low scores or have low weights.
    #   Run:
    #     1. Call /solve_step with fair_mode=False.
    #        Expected: Cell_B is chosen (due to Heuristic B's high weight * moderate score).
    #     2. Call /solve_step with fair_mode=True, min_weight_floor=0.1.
    #        Expected:
    #          - Heuristic A's score normalizes to ~1.0. Its effective contribution will be ~1.0 * max(0.01, 0.1) = 0.1.
    #          - Heuristic B's score normalizes to ~0.1 (if 100 was max). Its effective contribution will be ~0.1 * max(1.0, 0.1) = ~0.1.
    #          - Or, if 10 was high for M1 relative to its own scale, its normalized score might be higher.
    #          - The point is to see if Cell_A now has a chance or is chosen, demonstrating A2's "voice".
    #   Verification: Check `chosen_cells` in the response.

    # Test Case 2: All modules have zero configured weight, in Fair Mode
    #   Objective: Ensure all modules still contribute based on their normalized scores and the min_weight_floor.
    #   Setup:
    #     - Grid: Any suitable grid.
    #     - MODULE_WEIGHTS: Temporarily set all configured weights to 0.0.
    #   Run: Call /solve_step with fair_mode=True, min_weight_floor=0.1.
    #   Expected: The decision should be driven purely by which module produces the highest *normalized* score
    #             for some cell, as all will have an effective weight of 0.1.
    #   Verification: Analyze the `computed_scores_table` and `chosen_cells`. The cell with the highest
    #                 (sum of (normalized_score * 0.1)) should be chosen.

    # Test Case 3: Score Normalization Verification
    #   Objective: Verify that Min-Max normalization in fair_mode works correctly.
    #   Setup:
    #     - Grid: A grid where one heuristic (e.g., F5) produces a clear range of raw scores for empty cells
    #       (e.g., for 3 empty cells, scores [10, 60, 110]).
    #     - MODULE_WEIGHTS: Set weight of F5 to 1.0, others to 0.
    #   Run: Call /analyze_scores with fair_mode=True.
    #   Expected (Conceptual - need to inspect intermediate values or design heuristic carefully):
    #     - The raw scores [10, 60, 110] for F5, when normalized over these relevant cells,
    #       should become approximately [0.0, 0.5, 1.0].
    #     - The `computed_scores_table` will show the *final effective scores* after weighting.
    #       To truly test normalization, one might need to log the intermediate normalized map from
    #       `tensor_flow_score_vec_all` or mock the heuristic output.
    #   Verification: This is harder to verify directly from API output without more detailed logs.
    #                 One could craft a grid and a specific heuristic such that its normalized scores are predictable.

    # Test Case 4: Original Mode Integrity
    #   Objective: Verify that when fair_mode=False, the scoring is identical to the previous logic.
    #   Setup: Use any grid and weight configuration.
    #   Run:
    #     1. Call /solve_step with fair_mode=False.
    #     2. (If possible) Call the equivalent scoring logic from *before* fair_mode was introduced.
    #   Expected: The `computed_scores_table` and `chosen_cells` should be identical.
    #   Verification: Compare outputs.

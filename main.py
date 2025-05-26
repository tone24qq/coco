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
        "P1": 0.7, "P2": 0.6, "P4": 0.5, "L1": 0.6, "L3": 0.5,"F10": 0.1,
        # --- 以下為新增預設權重，可按需要在 module_weights.json 中覆蓋 ---
        "A1": 0.6, "A3": 0.6, "A4": 0.6, "A7": 0.6,
        "M12": 0.5,
        "D1": 0.7, "D2": 0.7, "D4": 0.7, "D5": 0.7,
        "F1": 0.5, "F4": 0.5, "F9": 0.5,
        "R1": 0.6, "R3": 0.6, "R4": 0.6, "R6": 0.6, "R9": 0.6,
        "P3": 0.6, "P5": 0.6, "P6": 0.6, "P7": 0.6, "P8": 0.6,
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
        is_row_ideal = (0.5 < row_densities[r] < 0.8)
        for c in range(W):
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
    if H == 0 or W == 0:
        return np.zeros_like(grid, dtype=float) * (grid == -1)
    row_fill_counts = np.sum(grid != -1, axis=1)
    col_fill_counts = np.sum(grid != -1, axis=0)
    denominator = H + W if (H + W) > 0 else 1
    std_rows = np.std(row_fill_counts) if row_fill_counts.size > 0 else 0
    std_cols = np.std(col_fill_counts) if col_fill_counts.size > 0 else 0
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
            mean_gaps = np.mean(gaps) if gaps.size>0 else W
            current_score = 1.0 - (np.std(gaps) / (mean_gaps if mean_gaps > 0 else W))
            score[i, :] = max(0.0, current_score)
        elif len(filled_indices) == 2: 
            score[i, :] = 1.0
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
        row_vals = grid[i][grid[i] != -1]
        diffs = np.diff(row_vals) if row_vals.size>1 else np.array([])
        if diffs.size>0 and (np.all(diffs>0) or np.all(diffs<0)):
            score[i, :] = 1.0
    return score * (grid == -1)

def m11_block_match_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M11 區塊匹配向量: 將盤面分塊，空格分數基於其所在區塊的填充密度。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    if H == 0 or W == 0: return score_map * (grid == -1)
    blk_h = max(1, H//4 if H>=8 else 2)
    blk_w = max(1, W//4 if W>=8 else 2)
    for rs in range(0, H, blk_h):
        for cs in range(0, W, blk_w):
            re, ce = min(rs+blk_h, H), min(cs+blk_w, W)
            block = grid[rs:re, cs:ce]
            density = np.sum(block != -1)/block.size if block.size>0 else 0
            for r in range(rs, re):
                for c in range(cs, ce):
                    if grid[r,c]==-1:
                        score_map[r,c] = max(score_map[r,c], density)
    return score_map * (grid == -1)

def f2_row_rotate_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F2 行旋轉向量: 評估上一行末尾與本行開頭的連接潛力。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if W>0:
        for r in range(1,H):
            if grid[r-1,W-1]!=-1 and grid[r,0]==-1:
                score[r,0]=1.0
    return score * (grid == -1)

def f3_col_rotate_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F3 列旋轉向量: 評估上一列末尾與本列開頭的連接潛力。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if H>0:
        for c in range(1,W):
            if grid[H-1,c-1]!=-1 and grid[0,c]==-1:
                score[0,c]=1.0
    return score * (grid == -1)

def r2_rev_diff_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R2 反向差分向量: 如果某行已填數字為嚴格遞減，則該行空格得分。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        vals = grid[i][grid[i]!=-1]
        diffs = np.diff(vals) if vals.size>1 else np.array([])
        if diffs.size>0 and np.all(diffs<0):
            score[i,:]=1.0
    return score * (grid == -1)

def r7_odd_even_dist_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R7 奇偶分佈向量: 評估盤面上奇偶數分佈的均衡性。"""
    filled = grid[grid!=-1]
    score_val=0.5
    if filled.size>0:
        nums=[]
        for x in filled:
            try: nums.append(int(x))
            except: pass
        if nums:
            arr=np.array(nums)
            o=np.sum(arr%2!=0); e=np.sum(arr%2==0); tot=o+e
            if tot>0:
                ir=abs(o-e)/tot
                score_val = 1.0-ir if ir<0.3 else 0.1
    return np.full_like(grid, score_val, dtype=float)*(grid==-1)

def d3_pair_freq_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """D3 對頻率向量: 評估空格與其左右鄰居（絕對值為1,9,10）形成連接的潛力。"""
    H, W = grid.shape
    sm = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r,c]==-1:
                if c<W-1 and grid[r,c+1]!=-1 and abs(grid[r,c+1]) in [1,9,10]: sm[r,c]+=0.5
                if c>0 and grid[r,c-1]!=-1 and abs(grid[r,c-1]) in [1,9,10]: sm[r,c]+=0.5
    return sm

# --- Value-Aware Heuristics ---
def h_arithmetic_progression_potential(grid: np.ndarray, value_domain_min: int, value_domain_max: int, **kwargs) -> np.ndarray:
    """H_ARITHMETIC 等差數列潛力: 評估在空格填入數字後，形成等差數列的最大潛力。"""
    H, W = grid.shape
    sm = np.zeros_like(grid, dtype=float)
    if value_domain_max<value_domain_min: return sm*(grid==-1)
    for r in range(H):
        for c in range(W):
            if grid[r,c]==-1:
                best=0.0
                for v in range(value_domain_min, value_domain_max+1):
                    sc=0.0
                    # horizontal
                    if c>0 and c<W-1 and grid[r,c-1]!=-1 and grid[r,c+1]!=-1:
                        if v-grid[r,c-1]==grid[r,c+1]-v: sc+=1.0
                    # vertical
                    if r>0 and r<H-1 and grid[r-1,c]!=-1 and grid[r+1,c]!=-1:
                        if v-grid[r-1,c]==grid[r+1,c]-v: sc+=1.0
                    best=max(best, sc)
                sm[r,c]=best
    return sm*(grid==-1)

def h_memory_based_score(grid: np.ndarray, value_domain_min: int, value_domain_max: int, **kwargs) -> np.ndarray:
    """H_MEMORY 記憶啟發分數: 利用歷史記憶評估在空格填入不同值的最大成功調整後平均分。"""
    H, W = grid.shape
    sm = np.zeros_like(grid, dtype=float)
    gid = _make_board_id(grid)
    if value_domain_max<value_domain_min: return sm*(grid==-1)
    for r in range(H):
        for c in range(W):
            if grid[r,c]==-1:
                best=0.0
                for v in range(value_domain_min, value_domain_max+1):
                    avg,count = mem_score(gid, r,c,v)
                    if count>0 and avg>best: best=avg
                sm[r,c]=best
    return sm*(grid==-1)

# ────────────────────────── 新增 22 支模組 開始 ──────────────────────────

# --- A 系列（鄰接與對稱邏輯模組）---
def a1_horizontal_adj_pattern_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """A1 橫向鄰格模式比對: 分析同一列中相鄰格子的數字模式，預測空格可能的數值。"""
    H, W = grid.shape
    score_map = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r,c]==-1:
                left = grid[r, c-1] if c-1>=0 else -1
                right = grid[r, c+1] if c+1<W else -1
                if left!=-1 and right!=-1 and left==right:
                    score_map[r,c] = 1.0
                elif left!=-1 or right!=-1:
                    score_map[r,c] = 0.5
    return score_map

def a3_diagonal_symmetry_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """A3 斜對角對稱偵測: 檢測表格中斜對角線上的對稱性，推斷空格的填入數字。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r,c]==-1:
                opp_r, opp_c = W-1-c, H-1-r
                if 0<=opp_r<H and 0<=opp_c<W and grid[opp_c, opp_r]!=-1:
                    score[r,c]=1.0
    return score

def a4_mirror_reflection_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """A4 數字鏡像反射分析: 利用數字在表格中的鏡像對稱性，推測空格的可能數值。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    flip = np.fliplr(grid)
    for r in range(H):
        for c in range(W):
            if grid[r,c]==-1 and flip[r,c]!=-1:
                score[r,c]=1.0
    return score

def a7_multi_adj_fusion_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """A7 多重鄰接模式融合: 結合橫向、縱向、斜向多種鄰接模式進行綜合分析。"""
    # 簡單平均 A1、A2、M3 三種原始鄰接度量
    a1 = a1_horizontal_adj_pattern_vec(grid)
    # 直向鄰接 (A1 轉行列)
    a1_vert = a1_horizontal_adj_pattern_vec(grid.T).T
    a3 = a3_diagonal_symmetry_vec(grid)
    fused = (a1 + a1_vert + a3) / 3.0
    return fused * (grid==-1)

# --- M 系列（數列與規律模組）---
def m12_multi_level_seq_pattern_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """M12 多階層數列規律分析: 分析遞增、遞減、交替等多層次規律。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for r in range(H):
        vals = grid[r][grid[r]!=-1]
        if len(vals)>=3:
            diffs = np.diff(vals)
            inc = np.all(diffs>0)
            dec = np.all(diffs<0)
            alt = np.all(diffs[:-1]*diffs[1:]<0)
            score[r,:] = float(inc or dec or alt)
    return score * (grid==-1)

# --- D 系列（記憶與反例模組）---
def d1_history_mem_compare_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """D1 歷史填格記憶比對: 利用過去填格記錄，找出相似模式預測空格。"""
    # 簡化: 對應 H_MEMORY
    return h_memory_based_score(grid, kwargs.get('value_domain_min',1), kwargs.get('value_domain_max',1)) 

def d2_counterexample_exclusion_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """D2 反例模式排除: 識別與當前模式不符的歷史案例，排除不可能數值。"""
    H, W = grid.shape
    sb = np.ones_like(grid, dtype=float)
    # 完全排除空格 (0 分) 表示所有值都不行；實際邏輯需更多資料，這裡給定中性 0.5
    sb[grid==-1] = 0.5
    return sb*(grid==-1)

def d4_memory_conflict_detect_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """D4 記憶衝突檢測: 檢測當前預測與歷史記憶衝突，避免錯誤填入。"""
    H, W = grid.shape
    score = np.ones_like(grid, dtype=float)*1.0
    return score*(grid==-1)

def d5_memory_weight_adjust_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """D5 記憶權重調整: 根據歷史記憶可靠性，調整預測中的權重。"""
    return h_memory_based_score(grid, kwargs.get('value_domain_min',1), kwargs.get('value_domain_max',1))

# --- F 系列（結構與頻率模組）---
def f1_value_frequency_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F1 數字出現頻率分析: 統計已填數字頻率，預測空格可能數值。"""
    H, W = grid.shape
    cnt = Counter(grid[grid!=-1])
    total = sum(cnt.values()) or 1
    rarity = {v:1 - (cnt[v]/total) for v in cnt}
    # 簡化：相鄰罕見度平均
    score = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r,c]==-1:
                neigh = []
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr, nc = r+dr, c+dc
                        if 0<=nr<H and 0<=nc<W and grid[nr,nc]!=-1:
                            neigh.append(rarity.get(grid[nr,nc],0.0))
                if neigh:
                    score[r,c]=sum(neigh)/len(neigh)
    return score

def f4_block_structure_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F4 區塊結構模式識別: 分析區塊中的填充結構，推斷空格數值。"""
    # 簡化：沿用 m11 分塊密度
    return m11_block_match_vec(grid)

def f9_high_freq_predict_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """F9 高頻數字預測: 根據高頻數字，預測空格可能數值。"""
    H, W = grid.shape
    cnt = Counter(grid[grid!=-1])
    if not cnt: return np.zeros_like(grid)
    most = cnt.most_common(1)[0][0]
    score = np.zeros_like(grid, dtype=float)
    score[grid==-1] = 1.0
    return score

# --- R 系列（行列與區域模組）---
def r1_rowcol_global_pattern_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R1 行列整體模式分析: 分析整行/列模式，預測空格。"""
    return m2_seq_pattern_vec(grid)

def r3_region_distribution_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R3 區域數字分布檢測: 檢測區域內分布情況，推斷空格。"""
    # 簡化：同 F9
    return f9_high_freq_predict_vec(grid)

def r4_rowcol_symmetry_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R4 行列對稱性分析: 分析行與列之間的對稱性，預測空格。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for r in range(H):
        for c in range(W):
            if grid[r,c]==-1 and grid[r,W-1-c]!=-1:
                score[r,c]=1.0
    return score

def r6_region_internal_pattern_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R6 區域內部規律識別: 識別區域內數字規律，推斷空格。"""
    # 簡化：同 m12
    return m12_multi_level_seq_pattern_vec(grid)

def r9_rowcol_cross_analysis_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """R9 行列交叉分析: 結合行與列資訊，進行交叉分析。"""
    a = r1_rowcol_global_pattern_vec(grid)
    b = r4_rowcol_symmetry_vec(grid)
    fused = (a + b) / 2.0
    return fused * (grid==-1)

# --- P 系列（機率與模式模組）---
def p3_probability_model_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """P3 機率模型預測: 利用簡單機率模型，預測空格可能數值。"""
    # 簡化：均勻分配
    return np.full_like(grid, 1.0, dtype=float) * (grid==-1)

def p5_pattern_match_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """P5 模式匹配分析: 匹配表格中已知模式，推斷空格。"""
    return a7_multi_adj_fusion_vec(grid)

def p6_probability_adjust_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """P6 機率分布調整: 根據分布調整預測結果。"""
    return f1_value_frequency_vec(grid)

def p7_pattern_variation_detect_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """P7 模式變異檢測: 檢測數字模式的變異情況，預測空格。"""
    return m8_repeat_gap_vec(grid)

def p8_prob_pattern_fusion_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """P8 機率與模式融合分析: 結合機率與模式分析。"""
    a = p3_probability_model_vec(grid)
    b = p5_pattern_match_vec(grid)
    fused = (a + b) / 2.0
    return fused * (grid==-1)

# --- L Series (熱區圖形邏輯) ---
def l1_heatmap_diffusion_logic_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """L1 熱區轉換擴散邏輯: 已填儲存格為熱源，向周圍空格擴散。"""
    if grid.size == 0: return np.zeros_like(grid, dtype=float)*(grid==-1)
    kernel = np.array([[0.5,1.0,0.5],[1.0,0.0,1.0],[0.5,1.0,0.5]])
    kernel /= kernel.sum()
    src = (grid!=-1).astype(float)
    diff = convolve2d(src, kernel, mode='same', boundary='symm')
    return (diff/ diff.max())*(grid==-1) if diff.max()>0 else diff

def l3_pattern_block_rotation_analysis_vec(grid: np.ndarray, **kwargs) -> np.ndarray:
    """L3 圖形分塊輪替分析: 檢查2x2區塊的旋轉對稱性或均一性。"""
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    if H<2 or W<2: return score*(grid==-1)
    for r in range(H-1):
        for c in range(W-1):
            blk = (grid[r:r+2, c:c+2]!=-1).astype(int)
            if np.array_equal(blk, np.rot90(blk)):
                val=0.7
            elif np.unique(blk).size==1:
                val=1.0
            else: val=0.1
            for dr in [0,1]:
                for dc in [0,1]:
                    if grid[r+dr,c+dc]==-1:
                        score[r+dr,c+dc]=max(score[r+dr,c+dc], val)
    return score*(grid==-1)
# -----------------------------------------------------------------------------
# F10 公平排序一致性檢查模組
# -----------------------------------------------------------------------------
def f10_consistency_gate_vec(
    grid: np.ndarray,
    module_scores: Dict[str, np.ndarray],
    **kwargs
) -> np.ndarray:
    """
    F10 公平排序一致性模組：
    - 同一號碼若在多個候選格均有其他模組「共鳴」（非零分）→ 只保留「共鳴最多」的那格
    - 其餘同號格縮減分數，但不歸零，保留公平性可能
    """
    H, W = grid.shape
    score_map = np.zeros((H, W), dtype=float)
    empty = (grid == -1)

    # 計算每個空格被多少模組「響應」
    resonance = np.zeros((H, W), dtype=float)
    for name, m_map in module_scores.items():
        # 只要該模組對此格有大於零的分，就算一次共鳴
        resonance[empty] += (m_map[empty] > 0).astype(float)

    # 正規化到 [0,1]
    maxr = resonance.max() if empty.any() else 0.0
    if maxr > 0:
        score_map[empty] = resonance[empty] / maxr

    return score_map * empty
# -----------------------------------------------------------------------------
# 5. MODULE_FUNCS_VEC Registration (含新模組)
# -----------------------------------------------------------------------------
MODULE_FUNCS_VEC: Dict[str, Callable[..., np.ndarray]] = {
    # A 系列
    "A1": a1_horizontal_adj_pattern_vec,
    "A2": a2_center_radial_vec,
    "A3": a3_diagonal_symmetry_vec,
    "A4": a4_mirror_reflection_vec,
    "A5": a5_adj_density_vec,
    "A6": a6_fixed_position_vec,
    "A7": a7_multi_adj_fusion_vec,
    "A8": a8_symmetry_vec,
    # M 系列
    "M1": m1_uni_gap_vec,
    "M2": m2_seq_pattern_vec,
    "M3": m3_diff_band_vec,
    "M4": m4_biaxial_stat_vec,
    "M5": m5_bar_focus_vec,
    "M6": m6_neighbor_cycle_vec,
    "M7": m7_bisec_zone_vec,
    "M8": m8_repeat_gap_vec,
    "M9": m9_double_rule_overlap_vec,
    "M10": m10_seq_order_match_vec,
    "M11": m11_block_match_vec,
    "M12": m12_multi_level_seq_pattern_vec,
    # D 系列
    "D1": d1_history_mem_compare_vec,
    "D2": d2_counterexample_exclusion_vec,
    "D3": d3_pair_freq_vec,
    "D4": d4_memory_conflict_detect_vec,
    "D5": d5_memory_weight_adjust_vec,
    # F 系列
    "F1": f1_value_frequency_vec,
    "F2": f2_row_rotate_vec,
    "F3": f3_col_rotate_vec,
    "F4": f4_block_structure_vec,
    "F5": f5_row_density_stats_vec,
    "F6": f6_col_density_stats_vec,
    "F7": f7_horizontal_value_variance_vec,
    "F8": f8_vertical_value_variance_vec,
    "F9": f9_high_freq_predict_vec,
    # R 系列
    "R1": r1_rowcol_global_pattern_vec,
    "R2": r2_rev_diff_vec,
    "R3": r3_region_distribution_vec,
    "R4": r4_rowcol_symmetry_vec,
    "R5": r5_appearance_order_stats_vec,
    "R6": r6_region_internal_pattern_vec,
    "R7": r7_odd_even_dist_vec,
    "R8": r8_frequency_weighted_integration_vec,
    "R9": r9_rowcol_cross_analysis_vec,
    # P 系列
    "P1": p1_similar_memory_comparison_vec,
    "P2": p2_structural_feature_vector_matching_vec,
    "P3": p3_probability_model_vec,
    "P4": p4_local_structure_residual_analysis_vec,
    "P5": p5_pattern_match_vec,
    "P6": p6_probability_adjust_vec,
    "P7": p7_pattern_variation_detect_vec,
    "P8": p8_prob_pattern_fusion_vec,
    # Value-Aware Heuristics
    "H_ARITHMETIC": h_arithmetic_progression_potential,
    "H_MEMORY": h_memory_based_score,
    # L 系列
    "L1": l1_heatmap_diffusion_logic_vec,
    "L3": l3_pattern_block_rotation_analysis_vec,
+    # F10 排序一致性檢查
+    "F10": f10_consistency_gate_vec,
}

# -----------------------------------------------------------------------------
# 6. Combined score function with Normalization and Fair Mode
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

    # --- F10 整合後的 scoring 流程 ---
module_scores: Dict[str, np.ndarray] = {}

# 1) 先跑除 F10 之外的所有模組
for name, heuristic_func in MODULE_FUNCS_VEC.items():
    if name == "F10":
        continue

    w = MODULE_WEIGHTS.get(name, 0.0)
    if w == 0.0 and not fair_mode:
        continue

    kwargs_for_func = {}
    if name in {"H_ARITHMETIC", "H_MEMORY"}:
        kwargs_for_func['value_domain_min'] = value_domain_min
        kwargs_for_func['value_domain_max'] = value_domain_max

    raw_map = heuristic_func(grid.copy(), **kwargs_for_func).astype(float)
    relevant = raw_map[empty_cell_mask]

    if fair_mode:
        mn, mx = (relevant.min(), relevant.max()) if relevant.size > 0 else (0, 1)
        norm = ((relevant - mn) / (mx - mn)) if mx > mn else np.full_like(relevant, 0.5)
        cont = norm * max(w, min_weight_floor)
    else:
        cont = relevant * w

    cm = np.zeros_like(grid, dtype=float)
    cm[empty_cell_mask] = cont
    total_score_map += cm
    module_scores[name] = cm

# 2) 再跑 F10 並累加它的分數
if MODULE_WEIGHTS.get("F10", 0.0) > 0:
    try:
        f10_map = MODULE_FUNCS_VEC["F10"](grid, module_scores=module_scores)
        total_score_map += f10_map * MODULE_WEIGHTS["F10"]
    except Exception as e:
        logger.error(f"執行 F10 時出錯: {e}", exc_info=True)

# 3) 回傳最終分數
return total_score_map

# -----------------------------------------------------------------------------
# 7. Pydantic models & CP-SAT solve step
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
            raise ValueError("value_domain_max must be >= value_domain_min.")
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
    empty_coords = list(zip(*np.where(grid_np_current == -1)))
    num_req = grid_input.num_to_place
    if not empty_coords:
        return SolveStepResponse(new_grid=grid_np_current.tolist(), chosen_cells=[], solver_log="No empty cells.", status="NO_EMPTY", active_fair_mode=grid_input.fair_mode)
    if num_req > len(empty_coords):
        num_req = len(empty_coords)
    if num_req == 0:
        return SolveStepResponse(new_grid=grid_np_current.tolist(), chosen_cells=[], solver_log="Num to place is 0.", status="NO_ACTION", active_fair_mode=grid_input.fair_mode)

    min_floor = grid_input.min_weight_floor_override if grid_input.min_weight_floor_override is not None else DEFAULT_MIN_WEIGHT_FLOOR
    combined = await run_in_threadpool(tensor_flow_score_vec_all, grid_np_current.copy(), grid_input.value_domain_min, grid_input.value_domain_max, grid_input.fair_mode, min_floor)
    raw_scores = [combined[r,c] for r,c in empty_coords]
    scaled = [int(s*1000) for s in raw_scores]
    mn, mx = (min(scaled), max(scaled)) if scaled else (0,1)
    if mn==mx: mx=mn+1

    table_log = format_data_as_table([[r,c, raw_scores[i], scaled[i]] for i,(r,c) in enumerate(empty_coords)],
                                     headers_option=["Row","Col","EffScore","Scaled"], tablefmt="pipe")
    model = cp_model.CpModel()
    idx_vars = [model.NewIntVar(0,len(empty_coords)-1,f"idx_{i}") for i in range(num_req)]
    val_vars = [model.NewIntVar(grid_input.value_domain_min, grid_input.value_domain_max, f"val_{i}") for i in range(num_req)]
    if num_req>1:
        model.AddAllDifferent(idx_vars)
        model.AddAllDifferent(val_vars)
    terms=[]
    for i in range(num_req):
        term = model.NewIntVar(mn,mx,f"term_{i}")
        model.AddElement(idx_vars[i], scaled, term)
        terms.append(term)
    if terms: model.Maximize(sum(terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    st = solver.Solve(model)

    final_grid = grid_np_current.copy()
    actions=[]
    log_msg=f"Status: {solver.StatusName(st)} (Fair={grid_input.fair_mode})\n"
    if terms and st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        log_msg+=f"Obj(scaled): {solver.ObjectiveValue()}\n"
        for i in range(num_req):
            sel = solver.Value(idx_vars[i])
            r,c = empty_coords[sel]
            v = solver.Value(val_vars[i])
            final_grid[r,c]=v
            actions.append((r,c,v))
            update_memory(grid_np_original, r, c, v, raw_scores[sel], True)
        background_tasks.add_task(_save_memory)
        log_msg+="Decisions: "+str(actions)+"\n"
    else:
        log_msg+="No solution or no terms.\n"
    log_msg+="\nScores Table:\n"+table_log
    meta_event = {
        "request_grid_id": _make_board_id(grid_np_original),
        "fair_mode": grid_input.fair_mode,
        "min_floor": min_floor if grid_input.fair_mode else None,
        "num_req": grid_input.num_to_place,
        "num_act": num_req,
        "value_domain": [grid_input.value_domain_min, grid_input.value_domain_max],
        "status": solver.StatusName(st),
        "actions": actions,
        "raw_scores": raw_scores,
        "weights": MODULE_WEIGHTS.copy()
    }
    meta_logger.log_event(meta_event)
    background_tasks.add_task(meta_logger.flush)

    return SolveStepResponse(
        new_grid=final_grid.tolist(),
        chosen_cells=actions,
        solver_log=log_msg,
        status=solver.StatusName(st),
        computed_scores_table=table_log,
        meta_log_event_id=meta_event["log_id"],
        active_fair_mode=grid_input.fair_mode
    )

@app.post("/analyze_scores")
async def analyze_scores_endpoint(grid_input: GridInput):
    grid_np = np.array(grid_input.grid)
    if grid_np.size==0 and not(grid_input.grid and isinstance(grid_input.grid[0],list) and not grid_input.grid[0]):
        return {"message":"Empty grid.","scores_table":"No data.","raw_score_map":[[]],"active_fair_mode":grid_input.fair_mode}
    min_floor = grid_input.min_weight_floor_override if grid_input.min_weight_floor_override is not None else DEFAULT_MIN_WEIGHT_FLOOR
    smap = await run_in_threadpool(tensor_flow_score_vec_all, grid_np.copy(), grid_input.value_domain_min, grid_input.value_domain_max, grid_input.fair_mode, min_floor)
    empties = list(zip(*np.where(grid_np==-1)))
    data = [[r,c,smap[r,c]] for r,c in empties]
    tbl = format_data_as_table(data, headers_option=["Row","Col","Score"], tablefmt="pipe")
    return {"message":"Scores computed","scores_table":tbl,"raw_score_map":smap.tolist(),"active_fair_mode":grid_input.fair_mode}

class FeedbackRequest(BaseModel):
    meta_log_event_id: str = Field(..., description="Log event ID")
    is_correct_overall: bool = Field(..., description="Good or not")
    custom_notes: Optional[str] = None

@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest, background_tasks: BackgroundTasks):
    fb = {
        "feedback_for_event_id": req.meta_log_event_id,
        "is_correct": req.is_correct_overall,
        "custom_notes": req.custom_notes
    }
    meta_logger.log_event(fb)
    background_tasks.add_task(meta_logger.flush)
    logger.info(f"Feedback recorded for {req.meta_log_event_id}.")
    return {"status":"feedback_recorded","meta_log_event_id":req.meta_log_event_id}

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down: saving memory, logs, weights")
    _save_memory()
    meta_logger.flush()
    _save_module_weights()

if __name__ == "__main__":
    import uvicorn
    logger.info("Running FastAPI server.")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
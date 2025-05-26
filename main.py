import logging
import math
import time
import os
import json
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod

import numpy as np
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from starlette.concurrency import run_in_threadpool
from numba import njit # Numba import

# ──────────────────────────────────────────────────────────────────────────────
# 0. Logging & Config
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Pydantic 模型定义
# ──────────────────────────────────────────────────────────────────────────────

class BoardInput(BaseModel):
    rows: int = Field(..., gt=0, description="盘面行数")
    cols: int = Field(..., gt=0, description="盘面列数")
    cells: List[List[Optional[int]]] = Field(
        ..., description="二维列表，None 表示空格"
    )
    logic_code_weights: Dict[str, float] = Field(
        ..., description="逻辑代码的基础权重，如 {'A1':0.8, ...}"
    )
    proposed_values: List[int] = Field(
        ..., description="要推理的目标数字列表"
    )
    active_modules: Optional[List[str]] = Field(
        None, description="要启用的模块ID列表，None或空表示启用所有模块"
    )
    module_weights: Optional[Dict[str, float]] = Field(
        None, description="模块贡献权重，覆盖全局设置"
    )
    top_n_count: int = Field(
        3, gt=0, description="每个目标数字返回的Top-N数量"
    )

    @validator("cells")
    def validate_cells_shape(cls, v, values):
        rows = values.get("rows")
        cols = values.get("cols")
        if not rows or not cols: # Should be caught by gt=0, but good for robustness
            return v
        if len(v) != rows or any(len(row) != cols for row in v):
            raise ValueError(f"cells 应为 {rows}×{cols} 的列表")
        return v

class PositionScore(BaseModel):
    position_code: str
    score: float
    confidence: Optional[float] = None

class ValuePrediction(BaseModel):
    proposed_value: int
    top_n_positions: List[PositionScore]

class InferenceResponse(BaseModel):
    predictions: List[ValuePrediction]
    processing_time_ms: Optional[float] = None
    warnings: Optional[List[str]] = None

class ModuleInfo(BaseModel):
    module_id: str
    name: str
    description: str

# ──────────────────────────────────────────────────────────────────────────────
# 2. 工具函数：逻辑代码生成
# ──────────────────────────────────────────────────────────────────────────────

def get_col_letter(n: int) -> str:
    s = ""
    while n >= 0:
        s = chr(ord("A") + (n % 26)) + s
        n = n // 26 - 1
    return s

def generate_logic_code(r: int, c: int) -> str:
    return f"{get_col_letter(c)}{r+1}"

# ──────────────────────────────────────────────────────────────────────────────
# 3. InternalBoardState：共享 NumPy 视图 + 假设查询
# ──────────────────────────────────────────────────────────────────────────────

class InternalBoardState:
    def __init__(self, src: BoardInput):
        self.src = src
        self.rows, self.cols = src.rows, src.cols

        arr = np.full((self.rows, self.cols), np.nan, dtype=np.float32)
        for r_idx in range(self.rows):
            for c_idx in range(self.cols):
                v = src.cells[r_idx][c_idx]
                if v is not None:
                    arr[r_idx, c_idx] = float(v) # Ensure float for consistency
        self._board: np.ndarray = arr
        self._fixed: np.ndarray = ~np.isnan(arr)
        self.fixed_values: set = {
            float(v) for row in src.cells for v in row if v is not None
        }
        self.active_modules: set = set(src.active_modules or [])

        # Cache for M1's max score to avoid recomputing it many times for the same request
        self._m1_max_logic_score = None


    def get_m1_max_logic_score(self) -> float:
        if self._m1_max_logic_score is None: # Calculate and cache if not already done
            all_positive_scores = [
                s for s in self.src.logic_code_weights.values() 
                if isinstance(s, (int, float)) and s > 0
            ]
            if not all_positive_scores:
                self._m1_max_logic_score = 0.0
            else:
                self._m1_max_logic_score = max(all_positive_scores)
        return self._m1_max_logic_score

    def is_fixed(self, r: int, c: int) -> bool:
        return self._fixed[r, c]

    def get_value_py(self, r: int, c: int, proposed: Optional[Tuple[int,int,float]] = None) -> float:
        if proposed and (r, c) == (proposed[0], proposed[1]):
            return proposed[2] # pv is already float
        return self._board[r, c]

    def logic_code(self, r: int, c: int) -> str:
        return generate_logic_code(r, c)

# ──────────────────────────────────────────────────────────────────────────────
# 4. 模块注册机制
# ──────────────────────────────────────────────────────────────────────────────

class LogicModule(ABC):
    module_id: str
    name: str
    description: str

    @abstractmethod
    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float: # pv changed to float
        ...

modules: Dict[str, LogicModule] = {}

def register_module(cls):
    inst = cls()
    modules[inst.module_id] = inst
    return cls

# ──────────────────────────────────────────────────────────────────────────────
# 5. M1: 基础得分模块 (NORMALIZED)
# ──────────────────────────────────────────────────────────────────────────────

@register_module
class M1_BaseScoreModule(LogicModule):
    module_id = "M1_BaseScore"
    name = "基础位置权重"
    description = "使用 logic_code_weights 提供的基础权重 (已归一化至0-1范围)"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        r, c = cell
        code = state.logic_code(r, c)
        
        current_raw_score = state.src.logic_code_weights.get(code, 0.0)
        if not isinstance(current_raw_score, (int, float)): # Ensure numeric
            current_raw_score = 0.0

        max_possible_score = state.get_m1_max_logic_score() # Use cached max score from state

        if max_possible_score > 0:
            if current_raw_score > 0: # Normalize only positive raw scores
                return current_raw_score / max_possible_score
            else: # Non-positive raw scores become 0
                return 0.0 
        else: # No positive scores in logic_code_weights at all
            return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 6. M4: 轴对称模块 (Numba Optimized Helper)
# ──────────────────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def _m4_analyze_numba(board: np.ndarray, rows: int, cols: int, r_proposed: int, c_proposed: int, pv_proposed: float) -> float:
    match_count = 0
    total_comparisons = 0

    def get_val(r_check: int, c_check: int) -> float:
        if r_check == r_proposed and c_check == c_proposed:
            return pv_proposed
        return board[r_check, c_check]

    # Horizontal symmetry check FOR THE PROPOSED ROW r_proposed
    for c_check in range(cols): # Iterate through all columns in the proposed row
        # If we are at the axis of symmetry for this pair, avoid double counting later (or ensure pair logic is correct)
        # For a cell (r_proposed, c_check), its symmetric partner is (r_proposed, cols - 1 - c_check)
        # Only compare if c_check <= (cols - 1 - c_check) to process each pair once
        if c_check > cols - 1 - c_check: 
            continue

        v1 = get_val(r_proposed, c_check)
        v2 = get_val(r_proposed, cols - 1 - c_check)

        if not np.isnan(v1) and not np.isnan(v2):
            if c_check == cols - 1 - c_check : # Element on the axis of symmetry
                 # A single element is symmetric with itself if it's part of the consideration
                 # The report's (source 201) `arr == np.fliplr(arr)` implies comparison of each cell with its counterpart.
                 # If an element is on the axis, it's compared with itself.
                 # The sum of (arr == flipped_arr) / count_of_non_nan_in_comparison
                 # Here, we sum matches and valid comparisons.
                match_count += 1 # Always matches itself
            elif v1 == v2: # Pair off-axis
                match_count += 2 # Counts for two cells matching
            total_comparisons += 2 if c_check != (cols - 1 - c_check) else 1
    
    # Vertical symmetry check FOR THE PROPOSED COLUMN c_proposed
    for r_check in range(rows):
        if r_check > rows - 1 - r_check:
            continue
        
        v1 = get_val(r_check, c_proposed)
        v2 = get_val(rows - 1 - r_check, c_proposed)
        if not np.isnan(v1) and not np.isnan(v2):
            if r_check == rows - 1 - r_check:
                match_count += 1
            elif v1 == v2:
                match_count += 2
            total_comparisons += 2 if r_check != (rows - 1 - r_check) else 1
            
    return float(match_count) / total_comparisons if total_comparisons > 0 else 0.0

@register_module
class M4_SymmetryAxialModule(LogicModule):
    module_id = "M4_SymmetryAxial"
    name = "轴对称性"
    description = "统计水平/垂直轴对称匹配率 (沿提案行列)"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        r, c = cell
        return _m4_analyze_numba(state._board, state.rows, state.cols, r, c, pv)

# ──────────────────────────────────────────────────────────────────────────────
# 7. M5: 段差分析模块 (Numba Optimized Helper)
# ──────────────────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def _m5_score_seq_numba(seq: np.ndarray) -> float:
    if seq.size < 3:
        return 0.0
    diffs = np.diff(seq)
    if diffs.size == 0: 
        return 0.0
    var = np.var(diffs) # Numba handles np.var
    return 1.0 / (1.0 + var) if (1.0 + var) != 0 else 0.0 # Avoid division by zero if var is -1 (unlikely for np.var)

@register_module
class M5_SegmentDiffModule(LogicModule):
    module_id = "M5_SegmentDiff"
    name = "段差分析"
    description = "行/列方向 diff 方差倒数"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        r, c = cell

        original_row_view = state._board[r, :]
        hypo_row = original_row_view.copy() 
        hypo_row[c] = pv
        seq_row = hypo_row[~np.isnan(hypo_row)]
        score_row = self._score_seq(seq_row)

        original_col_view = state._board[:, c]
        hypo_col = original_col_view.copy()
        hypo_col[r] = pv
        seq_col = hypo_col[~np.isnan(hypo_col)]
        score_col = self._score_seq(seq_col)
        
        return max(score_row, score_col)

    @staticmethod
    def _score_seq(seq: np.ndarray) -> float:
        return _m5_score_seq_numba(seq)

# ──────────────────────────────────────────────────────────────────────────────
# 8. InferenceEngine：统一加权聚合
# ──────────────────────────────────────────────────────────────────────────────

GLOBAL_MODULE_WEIGHTS = {
    "M1_BaseScore": 1.0,
    "M4_SymmetryAxial": 0.8,
    "M5_SegmentDiff": 0.7,
}

class InferenceEngine:
    def __init__(self):
        self.registry = modules

    def run_inference(self, state: InternalBoardState) -> Tuple[List[ValuePrediction], List[str]]:
        results: List[ValuePrediction] = []
        warnings: List[str] = []

        weights = GLOBAL_MODULE_WEIGHTS.copy()
        if state.src.module_weights:
            weights.update(state.src.module_weights)

        for pv_val_int in state.src.proposed_values: # pv from input is int
            pv_val_float = float(pv_val_int) # Convert to float for internal use
            if pv_val_float in state.fixed_values:
                warnings.append(f"Value {pv_val_int} 已存在于盘面，跳过。")
                results.append(ValuePrediction(proposed_value=pv_val_int, top_n_positions=[]))
                continue

            scores_for_pv: List[Tuple[str,float]] = []
            for r_idx in range(state.rows):
                for c_idx in range(state.cols):
                    if state.is_fixed(r_idx, c_idx):
                        continue
                    
                    aggregated_score = 0.0
                    total_weight = 0.0
                    current_cell = (r_idx, c_idx)

                    for mod_id, mod_instance in self.registry.items():
                        if state.active_modules and mod_id not in state.active_modules:
                            continue
                        
                        module_weight = weights.get(mod_id, 1.0)
                        individual_score = mod_instance.analyze(state, current_cell, pv_val_float) 
                        
                        aggregated_score += individual_score * module_weight
                        total_weight += module_weight
                    
                    final_cell_score = aggregated_score / total_weight if total_weight > 0 else 0.0
                    scores_for_pv.append((state.logic_code(r_idx,c_idx), final_cell_score))

            top_n = sorted(scores_for_pv, key=lambda x: x[1], reverse=True)[:state.src.top_n_count]
            position_scores = [PositionScore(position_code=code, score=round(sc,4)) for code, sc in top_n]
            results.append(ValuePrediction(proposed_value=pv_val_int, top_n_positions=position_scores))

        return results, warnings

# ──────────────────────────────────────────────────────────────────────────────
# 9. FastAPI App & 路由
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="自适应盘面推理系统", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# This instance will be shared if multiple requests come in quickly,
# so module-level caches need to be reset or handled per-call if state varies.
# The current M1 cache is on InternalBoardState, which is per-request.
engine_instance = InferenceEngine() 

def get_engine(): 
    return engine_instance

@app.post("/infer", response_model=InferenceResponse, summary="运行推理")
async def infer(board: BoardInput = Body(...), engine: InferenceEngine = Depends(get_engine)):
    start_time = time.perf_counter()
    try:
        internal_state = InternalBoardState(board) # New state for each request
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    predictions, warnings_list = await run_in_threadpool(engine.run_inference, internal_state)
    
    processing_duration_ms = (time.perf_counter() - start_time) * 1000
    return InferenceResponse(
        predictions=predictions, 
        processing_time_ms=round(processing_duration_ms, 2), 
        warnings=warnings_list if warnings_list else None
    )

@app.get("/config/modules", response_model=List[ModuleInfo], summary="可用模块列表")
def list_modules_info():
    return [ModuleInfo(module_id=m.module_id, name=m.name, description=m.description)
            for m in modules.values()]


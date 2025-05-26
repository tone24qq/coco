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

        # 构建 float32 数组，NaN 表示空
        arr = np.full((self.rows, self.cols), np.nan, dtype=np.float32)
        for r_idx in range(self.rows):
            for c_idx in range(self.cols):
                v = src.cells[r_idx][c_idx]
                if v is not None:
                    arr[r_idx, c_idx] = v
        self._board: np.ndarray = arr # Type hint for clarity
        # 固定格标记
        self._fixed: np.ndarray = ~np.isnan(arr) # Type hint for clarity
        # 已有数字集合
        self.fixed_values: set = {v for row in src.cells for v in row if v is not None}

        # 活跃模块集合
        self.active_modules: set = set(src.active_modules or [])

    def is_fixed(self, r: int, c: int) -> bool:
        return self._fixed[r, c]

    # This method is Python-based, Numba functions will need direct array access or specialized helpers
    def get_value_py(self, r: int, c: int, proposed: Optional[Tuple[int,int,int]] = None) -> float:
        if proposed and (r, c) == (proposed[0], proposed[1]):
            return float(proposed[2])
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
    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: int) -> float:
        ...

modules: Dict[str, LogicModule] = {}

def register_module(cls):
    inst = cls()
    modules[inst.module_id] = inst
    return cls

# ──────────────────────────────────────────────────────────────────────────────
# 5. M1: 基础得分模块
# ──────────────────────────────────────────────────────────────────────────────

@register_module
class M1_BaseScoreModule(LogicModule):
    module_id = "M1_BaseScore"
    name = "基础位置权重"
    description = "使用 logic_code_weights 提供的基础权重"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: int) -> float:
        r, c = cell
        code = state.logic_code(r, c)
        return state.src.logic_code_weights.get(code, 0.1)

# ──────────────────────────────────────────────────────────────────────────────
# 6. M4: 轴对称模块 (Numba Optimized Helper)
# ──────────────────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True) # Numba JIT compilation for performance
def _m4_analyze_numba(board: np.ndarray, rows: int, cols: int, r_proposed: int, c_proposed: int, pv_proposed: float) -> float:
    match_count = 0
    total_comparisons = 0

    # Helper function to get value considering the proposed move
    # This is inlined or made simple for Numba
    def get_val(r_check: int, c_check: int) -> float:
        if r_check == r_proposed and c_check == c_proposed:
            return pv_proposed
        return board[r_check, c_check]

    # Horizontal symmetry check (around the proposed cell's row)
    # This logic differs slightly from original, which checked symmetry for all rows/cols.
    # The report focuses on symmetry around the proposed placement.
    # Let's refine to check symmetry of the *entire board* if pv is placed.
    # For a given (r_current, c_current):
    #  v1 = value at (r_current, c_current) considering pv
    #  v2 = value at (r_current, cols - 1 - c_current) considering pv

    # Horizontal symmetry for the whole board with pv placed
    for r_iter in range(rows):
        for c_iter in range(cols // 2): # Iterate half, comparing to other half
            val1 = get_val(r_iter, c_iter)
            val2 = get_val(r_iter, cols - 1 - c_iter)

            if not np.isnan(val1) and not np.isnan(val2):
                if val1 == val2:
                    match_count += 1
                total_comparisons +=1
        # If odd number of columns, the middle column element is symmetric with itself if not NaN
        if cols % 2 == 1:
            val_mid = get_val(r_iter, cols // 2)
            if not np.isnan(val_mid):
                # match_count +=1 # A single element is always symmetric with itself
                total_comparisons +=1


    # Vertical symmetry for the whole board with pv placed
    for c_iter in range(cols):
        for r_iter in range(rows // 2): # Iterate half
            val1 = get_val(r_iter, c_iter)
            val2 = get_val(rows - 1 - r_iter, c_iter)

            if not np.isnan(val1) and not np.isnan(val2):
                if val1 == val2:
                    match_count += 1
                total_comparisons +=1
        # If odd number of rows, the middle row element is symmetric with itself
        if rows % 2 == 1:
            val_mid = get_val(rows // 2, c_iter)
            if not np.isnan(val_mid):
                # match_count +=1
                total_comparisons +=1

    # The original code sums matches from two separate full iterations (horizontal and vertical focused on proposed cell's line)
    # The report (source 201, 205-212) implies global symmetry after hypothetical placement.
    # The current Numba helper calculates global symmetry matches.
    # The previous Python code was:
    # # Horizontal (around proposed cell's row r)
    # for col_idx in range(state.cols):
    #     v1 = state.get_value_py(r_proposed, col_idx, (r_proposed, c_proposed, pv_proposed))
    #     v2 = state.get_value_py(r_proposed, state.cols - 1 - col_idx, (r_proposed, c_proposed, pv_proposed))
    #     if not np.isnan(v1) and not np.isnan(v2):
    #         match_count += (v1 == v2)
    #         total_comparisons += 1
    # # Vertical (around proposed cell's col c)
    # for row_idx in range(state.rows):
    #     v1 = state.get_value_py(row_idx, c_proposed, (r_proposed, c_proposed, pv_proposed))
    #     v2 = state.get_value_py(state.rows - 1 - row_idx, c_proposed, (r_proposed, c_proposed, pv_proposed))
    #     if not np.isnan(v1) and not np.isnan(v2):
    #         match_count += (v1 == v2)
    #         total_comparisons += 1
    # This interpretation (local symmetry lines) is simpler and closer to original `main.py`. Let's use that.

    # --- Re-implementing based on original main.py logic for M4 for closer parity ---
    match_count = 0
    total_comparisons = 0
    # Horizontal symmetry check FOR THE PROPOSED ROW r_proposed
    for c_check in range(cols):
        v1 = get_val(r_proposed, c_check)
        v2 = get_val(r_proposed, cols - 1 - c_check)
        if not np.isnan(v1) and not np.isnan(v2):
            if v1 == v2:
                match_count += 1
            total_comparisons += 1
    
    # Vertical symmetry check FOR THE PROPOSED COLUMN c_proposed
    for r_check in range(rows):
        v1 = get_val(r_check, c_proposed)
        v2 = get_val(rows - 1 - r_check, c_proposed)
        if not np.isnan(v1) and not np.isnan(v2):
            if v1 == v2:
                match_count += 1
            total_comparisons += 1
            
    return float(match_count) / total_comparisons if total_comparisons > 0 else 0.0


@register_module
class M4_SymmetryAxialModule(LogicModule):
    module_id = "M4_SymmetryAxial"
    name = "轴对称性"
    description = "统计水平/垂直轴对称匹配率"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: int) -> float:
        r, c = cell
        # Call the Numba-optimized helper function
        return _m4_analyze_numba(state._board, state.rows, state.cols, r, c, float(pv))

# ──────────────────────────────────────────────────────────────────────────────
# 7. M5: 段差分析模块 (Numba Optimized Helper)
# ──────────────────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True) # Numba JIT compilation for performance
def _m5_score_seq_numba(seq: np.ndarray) -> float:
    if seq.size < 3:
        return 0.0
    # Numba supports np.diff and np.var directly on NumPy arrays
    diffs = np.diff(seq)
    if diffs.size == 0: # Should not happen if seq.size >= 3, but defensive
        return 0.0
    var = np.var(diffs)
    return 1.0 / (1.0 + var)

@register_module
class M5_SegmentDiffModule(LogicModule):
    module_id = "M5_SegmentDiff"
    name = "段差分析"
    description = "行/列方向 diff 方差倒数"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: int) -> float:
        r, c = cell
        pv_float = float(pv)

        # 行分析
        # Extract the original row directly (it's a view, efficient)
        original_row_view = state._board[r, :]
        # Create a temporary copy FOR THE ROW ONLY to place the hypothetical pv
        hypo_row = original_row_view.copy() 
        hypo_row[c] = pv_float 
        # Extract non-NaN values (vectorized)
        seq_row = hypo_row[~np.isnan(hypo_row)]
        score_row = self._score_seq(seq_row)

        # 列分析
        original_col_view = state._board[:, c]
        hypo_col = original_col_view.copy()
        hypo_col[r] = pv_float
        seq_col = hypo_col[~np.isnan(hypo_col)]
        score_col = self._score_seq(seq_col)
        
        return max(score_row, score_col)

    @staticmethod
    # Link to the Numba-optimized static method
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

        for pv_val in state.src.proposed_values:
            if pv_val in state.fixed_values: # Efficient check
                warnings.append(f"Value {pv_val} 已存在于盘面，跳过。")
                results.append(ValuePrediction(proposed_value=pv_val, top_n_positions=[]))
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
                        # Critical: Module analyze methods are called here.
                        # Their performance directly impacts overall speed.
                        individual_score = mod_instance.analyze(state, current_cell, pv_val) 
                        
                        aggregated_score += individual_score * module_weight
                        total_weight += module_weight
                    
                    final_cell_score = aggregated_score / total_weight if total_weight > 0 else 0.0
                    scores_for_pv.append((state.logic_code(r_idx,c_idx), final_cell_score))

            top_n = sorted(scores_for_pv, key=lambda x: x[1], reverse=True)[:state.src.top_n_count]
            position_scores = [PositionScore(position_code=code, score=round(sc,4)) for code, sc in top_n]
            results.append(ValuePrediction(proposed_value=pv_val, top_n_positions=position_scores))

        return results, warnings

# ──────────────────────────────────────────────────────────────────────────────
# 9. FastAPI App & 路由
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="自适应盘面推理系统", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_engine(): 
    return InferenceEngine()

@app.post("/infer", response_model=InferenceResponse, summary="运行推理")
async def infer(board: BoardInput = Body(...), engine: InferenceEngine = Depends(get_engine)):
    start_time = time.perf_counter()
    try:
        # BoardInput validation happens here via Pydantic
        internal_state = InternalBoardState(board)
    except ValueError as e: # Catch Pydantic validation errors or others
        raise HTTPException(status_code=422, detail=str(e))
    
    # Offload CPU-bound task to thread pool
    predictions, warnings_list = await run_in_threadpool(engine.run_inference, internal_state)
    
    processing_duration_ms = (time.perf_counter() - start_time) * 1000
    return InferenceResponse(
        predictions=predictions, 
        processing_time_ms=round(processing_duration_ms, 2), 
        warnings=warnings_list if warnings_list else None
    )

@app.get("/config/modules", response_model=List[ModuleInfo], summary="可用模块列表")
def list_modules_info():
    console_debug_logger.debug("GET /config/modules request received.") # 调试日志
    return [ModuleInfo(module_id=m.module_id, name=m.name, description=m.description)
            for m in modules.values()]


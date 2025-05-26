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
        if not rows or not cols: 
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
                    arr[r_idx, c_idx] = float(v)
        self._board: np.ndarray = arr
        self._fixed: np.ndarray = ~np.isnan(arr)
        self.fixed_values: set = {
            float(v) for row in src.cells for v in row if v is not None
        }
        self.active_modules: set = set(src.active_modules or [])
        self._m1_max_logic_score = None


    def get_m1_max_logic_score(self) -> float:
        if self._m1_max_logic_score is None:
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
            return proposed[2]
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
    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
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
        if not isinstance(current_raw_score, (int, float)):
            current_raw_score = 0.0
        max_possible_score = state.get_m1_max_logic_score()
        if max_possible_score > 0:
            if current_raw_score > 0:
                return current_raw_score / max_possible_score
            else:
                return 0.0 
        else:
            return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 6. M4: 轴对称模块 (Numba Optimized Helper - GLOBAL SYMMETRY)
# ──────────────────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def _m4_analyze_numba_global_symmetry(board: np.ndarray, rows: int, cols: int, r_proposed: int, c_proposed: int, pv_proposed: float) -> float:
    
    # Helper to get value from board, considering the hypothetical placement
    def get_val(r_check: int, c_check: int) -> float:
        if r_check == r_proposed and c_check == c_proposed:
            return pv_proposed
        return board[r_check, c_check]

    h_matches = 0
    h_comparisons = 0
    # Global Horizontal symmetry: Compare each cell (r, c) with its horizontal counterpart (r, cols - 1 - c)
    # This method iterates over all cells and sums matches where both cell and its counterpart are numbers.
    for r_iter in range(rows):
        for c_iter in range(cols):
            val_orig = get_val(r_iter, c_iter)
            # Find the symmetric counterpart for val_orig
            # For horizontal symmetry, it's (r_iter, cols - 1 - c_iter)
            val_flipped = get_val(r_iter, cols - 1 - c_iter)
            
            if not np.isnan(val_orig) and not np.isnan(val_flipped):
                if val_orig == val_flipped:
                    h_matches += 1
                h_comparisons += 1
    # Normalize by the number of valid comparison points
    h_score = float(h_matches) / h_comparisons if h_comparisons > 0 else 0.0

    v_matches = 0
    v_comparisons = 0
    # Global Vertical symmetry: Compare each cell (r, c) with its vertical counterpart (rows - 1 - r, c)
    for r_iter in range(rows):
        for c_iter in range(cols):
            val_orig = get_val(r_iter, c_iter)
            # For vertical symmetry, it's (rows - 1 - r_iter, c_iter)
            val_flipped = get_val(rows - 1 - r_iter, c_iter)

            if not np.isnan(val_orig) and not np.isnan(val_flipped):
                if val_orig == val_flipped:
                    v_matches += 1
                v_comparisons += 1
    v_score = float(v_matches) / v_comparisons if v_comparisons > 0 else 0.0
    
    # Combine scores: average the horizontal and vertical symmetry scores.
    # If one dimension has no valid comparisons (e.g., a 1x1 board for one of the sub-calcs, though unlikely here),
    # rely on the other, or if both are valid, average them.
    if h_comparisons > 0 and v_comparisons > 0:
      # If board is 1 row, v_score will be 1.0. If 1 col, h_score will be 1.0.
      # This seems acceptable as it reflects perfect symmetry along the degenerate axis.
      return (h_score + v_score) / 2.0
    elif h_comparisons > 0: # e.g. single row board
      return h_score
    elif v_comparisons > 0: # e.g. single col board
      return v_score
    return 0.0 # No valid comparisons possible

@register_module
class M4_SymmetryAxialModule(LogicModule):
    module_id = "M4_SymmetryAxial"
    name = "全局轴对称性" # Updated name to reflect change
    description = "评估整个盘面在假设落子后的全局水平和垂直对称性" # Updated description

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        r, c = cell
        # Call the Numba-optimized helper function for global symmetry
        return _m4_analyze_numba_global_symmetry(state._board, state.rows, state.cols, r, c, pv)

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
    var = np.var(diffs)
    denominator = 1.0 + var
    return 1.0 / denominator if denominator != 0 else 0.0 # Avoid division by zero for safety

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
    "M4_SymmetryAxial": 0.8, # Weight for the new Global Symmetry module
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

        for pv_val_int in state.src.proposed_values:
            pv_val_float = float(pv_val_int)
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

                    active_module_keys = state.active_modules
                    # If active_modules is empty in state, it means use all from registry
                    # Otherwise, use only the ones specified in active_modules
                    modules_to_run = self.registry.items()
                    if active_module_keys: # if set is not empty
                        modules_to_run = [(mid, mod) for mid, mod in self.registry.items() if mid in active_module_keys]
                    
                    if not modules_to_run and self.registry:
                        # This case implies active_modules was specified but none matched registered modules.
                        # Or, if active_modules was empty, but registry is also empty (though unlikely).
                        # For safety, if no modules are to be run for a cell, score is 0.
                        # However, the outer logic for active_modules in InternalBoardState (empty set means use all)
                        # means this check might be more about an empty registry.
                        # The current logic iterates `modules_to_run`. If it's empty, loop doesn't run, score remains 0.
                        pass


                    for mod_id, mod_instance in modules_to_run:
                        # The check `if state.active_modules and mid not in state.active_modules:`
                        # is implicitly handled by how `modules_to_run` is constructed.
                        
                        module_weight = weights.get(mod_id, 1.0) # Default weight if not in custom weights
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

engine_instance = InferenceEngine() 

def get_engine(): 
    return engine_instance

@app.post("/infer", response_model=InferenceResponse, summary="运行推理")
async def infer(board: BoardInput = Body(...), engine: InferenceEngine = Depends(get_engine)):
    start_time = time.perf_counter()
    try:
        internal_state = InternalBoardState(board)
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


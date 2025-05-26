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
        None, description="要启用的模块ID列表，None或空表示启用所有已注册模块"
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
    # Optional: to include detailed contributions if needed for debugging
    # detail_contributions: Optional[Dict[str, List[Dict[str, Any]]]] = None 

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
        # If active_modules is None or empty list in input, this will be an empty set.
        # InferenceEngine will interpret empty set as "run all registered modules".
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
    logger.info(f"Registered module: {inst.module_id} - {inst.name}")
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
# 6. M4: 轴对称模块 (GLOBAL SYMMETRY) - Corresponds to user's "A6"
# ──────────────────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def _m4_analyze_numba_global_symmetry(board: np.ndarray, rows: int, cols: int, r_proposed: int, c_proposed: int, pv_proposed: float) -> float:
    def get_val(r_check: int, c_check: int) -> float:
        if r_check == r_proposed and c_check == c_proposed:
            return pv_proposed
        return board[r_check, c_check]

    h_matches = 0
    h_comparisons = 0
    for r_iter in range(rows):
        for c_iter in range(cols):
            val_orig = get_val(r_iter, c_iter)
            val_flipped = get_val(r_iter, cols - 1 - c_iter)
            if not np.isnan(val_orig) and not np.isnan(val_flipped):
                if val_orig == val_flipped:
                    h_matches += 1
                h_comparisons += 1
    h_score = float(h_matches) / h_comparisons if h_comparisons > 0 else 0.0

    v_matches = 0
    v_comparisons = 0
    for r_iter in range(rows):
        for c_iter in range(cols):
            val_orig = get_val(r_iter, c_iter)
            val_flipped = get_val(rows - 1 - r_iter, c_iter)
            if not np.isnan(val_orig) and not np.isnan(val_flipped):
                if val_orig == val_flipped:
                    v_matches += 1
                v_comparisons += 1
    v_score = float(v_matches) / v_comparisons if v_comparisons > 0 else 0.0
    
    if h_comparisons > 0 and v_comparisons > 0:
      return (h_score + v_score) / 2.0
    elif h_comparisons > 0:
      return h_score
    elif v_comparisons > 0:
      return v_score
    return 0.0

@register_module
class M4_SymmetryAxialModule(LogicModule): # This is user's "A6"
    module_id = "M4_SymmetryAxial"
    name = "全局轴对称性 (A6)"
    description = "评估整个盘面在假设落子后的全局水平和垂直对称性"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        r, c = cell
        return _m4_analyze_numba_global_symmetry(state._board, state.rows, state.cols, r, c, pv)

# ──────────────────────────────────────────────────────────────────────────────
# 7. M5: 段差分析模块
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
    return 1.0 / denominator if denominator != 0 else 0.0

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
# 8. 新增占位符模块 (A5, M2, M8, R1, R2)
# ──────────────────────────────────────────────────────────────────────────────

@register_module
class A5_PlaceholderModule(LogicModule):
    module_id = "A5_Placeholder"
    name = "A5 占位符模块"
    description = "这是一个A5模块的占位符实现，返回固定值0.5。"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        # 未来实现具体逻辑
        return 0.5 # Dummy score

@register_module
class M2_PlaceholderModule(LogicModule):
    module_id = "M2_Placeholder"
    name = "M2 占位符模块"
    description = "这是一个M2模块的占位符实现，返回固定值0.3。"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        # 未来实现具体逻辑
        return 0.3 # Dummy score

@register_module
class M8_PlaceholderModule(LogicModule):
    module_id = "M8_Placeholder"
    name = "M8 占位符模块"
    description = "这是一个M8模块的占位符实现，返回固定值0.4。"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        # 未来实现具体逻辑
        return 0.4 # Dummy score

@register_module
class R1_SequenceRuleModule(LogicModule):
    module_id = "R1_SequenceRule"
    name = "R1 序列规则模块"
    description = "R1模块，检查特定序列规则的占位符，返回固定值0.6。"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        # 未来实现具体逻辑
        # 例如: 检查 (r,c) 周围是否形成特定序列 (等差、等比等)
        return 0.6 # Dummy score

@register_module
class R2_RelativePositionModule(LogicModule):
    module_id = "R2_RelativePosition"
    name = "R2 相对位置模块"
    description = "R2模块，评估与现有数字相对位置关系的占位符，返回固定值0.2。"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        # 未来实现具体逻辑
        # 例如: 检查 pv 与 (r,c) 上下左右固定数字的关系
        return 0.2 # Dummy score

# ──────────────────────────────────────────────────────────────────────────────
# 9. InferenceEngine：统一加权聚合
# ──────────────────────────────────────────────────────────────────────────────

GLOBAL_MODULE_WEIGHTS = {
    "M1_BaseScore": 1.0,
    "M4_SymmetryAxial": 0.8, # A6
    "M5_SegmentDiff": 0.7,
    "A5_Placeholder": 0.5,   # Default weight for new module
    "M2_Placeholder": 0.5,   # Default weight for new module
    "M8_Placeholder": 0.5,   # Default weight for new module
    "R1_SequenceRule": 0.6,  # Default weight for new module
    "R2_RelativePosition": 0.4, # Default weight for new module
}

class InferenceEngine:
    def __init__(self):
        self.registry = modules # Populated by @register_module

    def run_inference(self, state: InternalBoardState) -> Tuple[List[ValuePrediction], List[str]]:
        results: List[ValuePrediction] = []
        warnings: List[str] = []
        # For detailed contributions, if you add it to InferenceResponse
        # detail_contributions_response: Dict[str, List[Dict[str, Any]]] = {} 

        effective_weights = GLOBAL_MODULE_WEIGHTS.copy()
        if state.src.module_weights: # Allow request to override global weights
            effective_weights.update(state.src.module_weights)

        for pv_val_int in state.src.proposed_values:
            pv_val_float = float(pv_val_int)
            if pv_val_float in state.fixed_values:
                warnings.append(f"Value {pv_val_int} 已存在于盘面，跳过。")
                results.append(ValuePrediction(proposed_value=pv_val_int, top_n_positions=[]))
                continue

            scores_for_pv: List[Tuple[str,float]] = []
            # For detailed contributions
            # current_pv_contributions: List[Dict[str, Any]] = []

            for r_idx in range(state.rows):
                for c_idx in range(state.cols):
                    if state.is_fixed(r_idx, c_idx):
                        continue
                    
                    aggregated_score = 0.0
                    total_weight = 0.0
                    current_cell = (r_idx, c_idx)
                    position_code = state.logic_code(r_idx,c_idx)

                    # Determine which modules to run
                    modules_to_iterate: List[Tuple[str, LogicModule]]
                    if not state.active_modules: # Empty set means run all registered modules
                        modules_to_iterate = list(self.registry.items())
                    else:
                        modules_to_iterate = [
                            (mid, mod) for mid, mod in self.registry.items() 
                            if mid in state.active_modules
                        ]
                    
                    # For detailed contributions for this cell
                    # cell_contributions_detail: List[Dict[str, Any]] = []

                    for mod_id, mod_instance in modules_to_iterate:
                        module_weight = effective_weights.get(mod_id, 1.0) # Default weight is 1.0 if not specified
                        
                        individual_score = mod_instance.analyze(state, current_cell, pv_val_float)
                        
                        weighted_score = individual_score * module_weight
                        aggregated_score += weighted_score
                        total_weight += module_weight
                        
                        # For detailed contributions
                        # cell_contributions_detail.append({
                        #     "module_id": mod_id,
                        #     "raw_score": round(individual_score, 4),
                        #     "weight": round(module_weight, 4),
                        #     "weighted_score": round(weighted_score, 4)
                        # })
                    
                    final_cell_score = aggregated_score / total_weight if total_weight > 0 else 0.0
                    scores_for_pv.append((position_code, final_cell_score))
                    
                    # For detailed contributions
                    # if cell_contributions_detail: # Only add if there were contributions
                    #    current_pv_contributions.append({
                    #        "position_code": position_code,
                    #        "final_score": round(final_cell_score, 4),
                    #        "module_breakdown": cell_contributions_detail
                    #    })

            # For detailed contributions
            # if current_pv_contributions:
            #    detail_contributions_response[str(pv_val_int)] = current_pv_contributions

            top_n = sorted(scores_for_pv, key=lambda x: x[1], reverse=True)[:state.src.top_n_count]
            position_scores = [PositionScore(position_code=code, score=round(sc,4)) for code, sc in top_n]
            results.append(ValuePrediction(proposed_value=pv_val_int, top_n_positions=position_scores))

        # If returning detailed contributions:
        # return results, warnings, detail_contributions_response 
        return results, warnings

# ──────────────────────────────────────────────────────────────────────────────
# 10. FastAPI App & 路由
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
    
    # Modify if InferenceEngine returns more data like detail_contributions
    predictions, warnings_list = await run_in_threadpool(engine.run_inference, internal_state)
    
    processing_duration_ms = (time.perf_counter() - start_time) * 1000
    
    response_data = {
        "predictions": predictions,
        "processing_time_ms": round(processing_duration_ms, 2),
        "warnings": warnings_list if warnings_list else None,
        # "detail_contributions": detail_contributions # if you enable this
    }
    return InferenceResponse(**response_data)

@app.get("/config/modules", response_model=List[ModuleInfo], summary="可用模块列表")
def list_modules_info():
    return [ModuleInfo(module_id=m.module_id, name=m.name, description=m.description)
            for m in modules.values()]


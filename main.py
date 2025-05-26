import logging
import math
import time
import os
import json
from typing import List, Optional, Dict, Tuple, Any # Added Any for detailed log
from abc import ABC, abstractmethod

import numpy as np
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from starlette.concurrency import run_in_threadpool
from numba import njit

# ──────────────────────────────────────────────────────────────────────────────
# 0. Logging & Config
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Add a handler for direct print-like logging to console for debugging
console_debug_logger = logging.getLogger("console_debug")
console_debug_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s DEBUG: %(message)s'))
console_debug_logger.addHandler(console_handler)
console_debug_logger.propagate = False


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
        console_debug_logger.debug(f"InternalBoardState: Initialized active_modules set: {self.active_modules if self.active_modules else 'EMPTY (run all)'}")
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
    logger.info(f"Registered module: {inst.module_id} - {inst.name}") # Standard logger
    console_debug_logger.debug(f"DEBUG REGISTER: Module '{inst.module_id}' registered.") # Debug logger
    return cls

# ──────────────────────────────────────────────────────────────────────────────
# 5. M1: 基础得分模块 (NORMALIZED - DEBUG FIXED SCORE)
# ──────────────────────────────────────────────────────────────────────────────

@register_module
class M1_BaseScoreModule(LogicModule):
    module_id = "M1_BaseScore"
    name = "基础位置权重"
    description = "DEBUG: 固定返回 0.111"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        return 0.111 # DEBUG: Fixed score

# ──────────────────────────────────────────────────────────────────────────────
# 6. M4: 轴对称模块 (DEBUG FIXED SCORE) - Corresponds to user's "A6"
# ──────────────────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True) # Numba JIT for the helper
def _m4_analyze_numba_DEBUG(board: np.ndarray, rows: int, cols: int, r_proposed: int, c_proposed: int, pv_proposed: float) -> float:
    # Actual logic commented out for debug
    # ... (original _m4_analyze_numba_global_symmetry logic can be here) ...
    return 0.444 # DEBUG: Fixed score for M4 ("A6")

@register_module
class M4_SymmetryAxialModule(LogicModule): # This is user's "A6"
    module_id = "M4_SymmetryAxial"
    name = "全局轴对称性 (A6)"
    description = "DEBUG: 固定返回 0.444"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        r, c = cell
        # Pass necessary args for Numba function, even if not used by the debug version
        return _m4_analyze_numba_DEBUG(state._board, state.rows, state.cols, r, c, pv)


# ──────────────────────────────────────────────────────────────────────────────
# 7. M5: 段差分析模块 (DEBUG FIXED SCORE)
# ──────────────────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True) # Numba JIT for the helper
def _m5_score_seq_numba_DEBUG(seq: np.ndarray) -> float:
    # Actual logic commented out for debug
    # ... (original _m5_score_seq_numba logic) ...
    return 0.555 # DEBUG: Fixed score for M5's core logic

@register_module
class M5_SegmentDiffModule(LogicModule):
    module_id = "M5_SegmentDiff"
    name = "段差分析"
    description = "DEBUG: 固定返回 0.555"

    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float:
        # The sequence extraction logic is kept, but the final scoring uses the debug Numba function
        r, c = cell
        original_row_view = state._board[r, :] # This is a view
        # hypo_row = original_row_view.copy() # Not needed if score is fixed
        # hypo_row[c] = pv
        # seq_row = hypo_row[~np.isnan(hypo_row)]
        # For debug, we can just pass a dummy array to the Numba func if its inputs are truly ignored
        dummy_seq = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        return _m5_score_seq_numba_DEBUG(dummy_seq) # Call debug version

    @staticmethod
    def _score_seq(seq: np.ndarray) -> float: # This method is now bypassed in analyze for fixed score
        return _m5_score_seq_numba_DEBUG(seq)

# ──────────────────────────────────────────────────────────────────────────────
# 8. 新增占位符模块 (A5, M2, M8, R1, R2) - Already return fixed scores
# ──────────────────────────────────────────────────────────────────────────────

@register_module
class A5_PlaceholderModule(LogicModule):
    module_id = "A5_Placeholder"
    name = "A5 占位符模块"
    description = "这是一个A5模块的占位符实现，返回固定值0.5。"
    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float: return 0.5

@register_module
class M2_PlaceholderModule(LogicModule):
    module_id = "M2_Placeholder"
    name = "M2 占位符模块"
    description = "这是一个M2模块的占位符实现，返回固定值0.3。"
    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float: return 0.3

@register_module
class M8_PlaceholderModule(LogicModule):
    module_id = "M8_Placeholder"
    name = "M8 占位符模块"
    description = "这是一个M8模块的占位符实现，返回固定值0.4。"
    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float: return 0.4

@register_module
class R1_SequenceRuleModule(LogicModule):
    module_id = "R1_SequenceRule"
    name = "R1 序列规则模块"
    description = "R1模块，检查特定序列规则的占位符，返回固定值0.6。"
    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float: return 0.6

@register_module
class R2_RelativePositionModule(LogicModule):
    module_id = "R2_RelativePosition"
    name = "R2 相对位置模块"
    description = "R2模块，评估与现有数字相对位置关系的占位符，返回固定值0.2。"
    def analyze(self, state: InternalBoardState, cell: Tuple[int,int], pv: float) -> float: return 0.2

# ──────────────────────────────────────────────────────────────────────────────
# 9. InferenceEngine：统一加权聚合 (WITH EXTENSIVE DEBUG LOGGING)
# ──────────────────────────────────────────────────────────────────────────────

GLOBAL_MODULE_WEIGHTS = {
    "M1_BaseScore": 1.0,
    "M4_SymmetryAxial": 1.0, # "A6" - Weight for debug
    "M5_SegmentDiff": 1.0,   # Weight for debug
    "A5_Placeholder": 1.0,
    "M2_Placeholder": 1.0,
    "M8_Placeholder": 1.0,
    "R1_SequenceRule": 1.0,
    "R2_RelativePosition": 1.0,
} # Set all weights to 1.0 for easy debug sum

class InferenceEngine:
    def __init__(self):
        self.registry = modules
        console_debug_logger.debug(f"InferenceEngine initialized. Registered modules: {list(self.registry.keys())}")

    def run_inference(self, state: InternalBoardState) -> Tuple[List[ValuePrediction], List[str]]:
        console_debug_logger.debug(f"--- Starting run_inference for PVs: {state.src.proposed_values} ---")
        results: List[ValuePrediction] = []
        warnings: List[str] = []
        effective_weights = GLOBAL_MODULE_WEIGHTS.copy()
        if state.src.module_weights: # Allow request to override global weights
            console_debug_logger.debug(f"Overriding global weights with request module_weights: {state.src.module_weights}")
            effective_weights.update(state.src.module_weights)
        console_debug_logger.debug(f"Effective module weights being used: {effective_weights}")


        for pv_val_int in state.src.proposed_values:
            pv_val_float = float(pv_val_int)
            console_debug_logger.debug(f"Processing Proposed Value (PV): {pv_val_float}")
            if pv_val_float in state.fixed_values:
                warnings.append(f"Value {pv_val_int} 已存在于盘面，跳过。")
                console_debug_logger.debug(f"PV {pv_val_float} exists on board. Skipping.")
                results.append(ValuePrediction(proposed_value=pv_val_int, top_n_positions=[]))
                continue

            scores_for_pv: List[Tuple[str,float]] = []
            for r_idx in range(state.rows):
                for c_idx in range(state.cols):
                    if state.is_fixed(r_idx, c_idx):
                        continue
                    
                    current_cell = (r_idx, c_idx)
                    position_code = state.logic_code(r_idx,c_idx)
                    console_debug_logger.debug(f"  Calculating score for Cell: {position_code} ({current_cell}) with PV: {pv_val_float}")

                    aggregated_score = 0.0
                    total_weight = 0.0
                    
                    modules_to_iterate: List[Tuple[str, LogicModule]]
                    if not state.active_modules: 
                        modules_to_iterate = list(self.registry.items())
                        console_debug_logger.debug(f"    state.active_modules is EMPTY, running ALL {len(modules_to_iterate)} registered modules.")
                    else:
                        modules_to_iterate = [
                            (mid, mod) for mid, mod in self.registry.items() 
                            if mid in state.active_modules
                        ]
                        console_debug_logger.debug(f"    state.active_modules IS SET ({state.active_modules}), running {len(modules_to_iterate)} filtered modules.")
                    
                    if not modules_to_iterate:
                        console_debug_logger.debug(f"    No modules to iterate for cell {position_code}. This is unusual if registry is not empty.")

                    for mod_id, mod_instance in modules_to_iterate:
                        console_debug_logger.debug(f"      Considering Module: {mod_id}")
                        module_weight = effective_weights.get(mod_id, 1.0) # Default weight is 1.0
                        
                        individual_score = mod_instance.analyze(state, current_cell, pv_val_float) 
                        console_debug_logger.debug(f"        Module '{mod_id}': analyze() returned raw_score = {individual_score:.3f}")
                        
                        weighted_score = individual_score * module_weight
                        aggregated_score += weighted_score
                        total_weight += module_weight
                        console_debug_logger.debug(f"        Module '{mod_id}': weight = {module_weight:.2f}, weighted_score = {weighted_score:.3f}")
                        console_debug_logger.debug(f"        Running totals: aggregated_score = {aggregated_score:.3f}, total_weight = {total_weight:.2f}")
                    
                    final_cell_score = aggregated_score / total_weight if total_weight > 0 else 0.0
                    console_debug_logger.debug(f"    Score for Cell {position_code} ({current_cell}), PV {pv_val_float}: {final_cell_score:.4f}\n")
                    scores_for_pv.append((position_code, final_cell_score))
            
            top_n = sorted(scores_for_pv, key=lambda x: x[1], reverse=True)[:state.src.top_n_count]
            position_scores = [PositionScore(position_code=code, score=round(sc,4)) for code, sc in top_n]
            results.append(ValuePrediction(proposed_value=pv_val_int, top_n_positions=position_scores))
        console_debug_logger.debug(f"--- Finished run_inference for PVs: {state.src.proposed_values} ---")
        return results, warnings

# ──────────────────────────────────────────────────────────────────────────────
# 10. FastAPI App & 路由
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="自适应盘面推理系统 (DEBUG MODE)", version="1.0-debug")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

engine_instance = InferenceEngine() 

def get_engine(): 
    return engine_instance

@app.post("/infer", response_model=InferenceResponse, summary="运行推理")
async def infer(board: BoardInput = Body(...), engine: InferenceEngine = Depends(get_engine)):
    start_time = time.perf_counter()
    console_debug_logger.debug(f"\n\n\n=== New /infer request received ===")
    console_debug_logger.debug(f"Request BoardInput: active_modules='{board.active_modules}'") # Log input active_modules
    try:
        internal_state = InternalBoardState(board)
    except ValueError as e:
        console_debug_logger.error(f"Error creating InternalBoardState: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))
    
    predictions, warnings_list = await run_in_threadpool(engine.run_inference, internal_state)
    
    processing_duration_ms = (time.perf_counter() - start_time) * 1000
    
    response_data = {
        "predictions": predictions,
        "processing_time_ms": round(processing_duration_ms, 2),
        "warnings": warnings_list if warnings_list else None,
    }
    console_debug_logger.debug(f"=== Finished /infer request. Processing time: {processing_duration_ms:.2f}ms ===\n")
    return InferenceResponse(**response_data)

@app.get("/config/modules", response_model=List[ModuleInfo], summary="可用模块列表")
def list_modules_info():
    console_debug_logger.debug("GET /config/modules request received.")
    return [ModuleInfo(module_id=m.module_id, name=m.name, description=m.description)
            for m in modules.values()]


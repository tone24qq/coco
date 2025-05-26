# main.py

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
        for r in range(self.rows):
            for c in range(self.cols):
                v = src.cells[r][c]
                if v is not None:
                    arr[r, c] = v
        self._board = arr
        # 固定格标记
        self._fixed = ~np.isnan(arr)
        # 已有数字集合
        self.fixed_values = {v for row in src.cells for v in row if v is not None}

        # 活跃模块集合
        self.active_modules = set(src.active_modules or [])

    def is_fixed(self, r: int, c: int) -> bool:
        return self._fixed[r, c]

    def get_value(self, r: int, c: int, proposed: Optional[Tuple[int,int,int]] = None) -> float:
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

    def analyze(self, state, cell, pv):
        r, c = cell
        code = state.logic_code(r, c)
        return state.src.logic_code_weights.get(code, 0.1)

# ──────────────────────────────────────────────────────────────────────────────
# 6. M4: 轴对称模块
# ──────────────────────────────────────────────────────────────────────────────

@register_module
class M4_SymmetryAxialModule(LogicModule):
    module_id = "M4_SymmetryAxial"
    name = "轴对称性"
    description = "统计水平/垂直轴对称匹配率"

    def analyze(self, state, cell, pv):
        r, c = cell
        match = 0; total = 0
        # 水平
        for col in range(state.cols):
            v1 = state.get_value(r, col, (*cell, pv))
            v2 = state.get_value(r, state.cols-1-col, (*cell, pv))
            if not np.isnan(v1) and not np.isnan(v2):
                match += (v1 == v2)
                total += 1
        # 垂直
        for row in range(state.rows):
            v1 = state.get_value(row, c, (*cell, pv))
            v2 = state.get_value(state.rows-1-row, c, (*cell, pv))
            if not np.isnan(v1) and not np.isnan(v2):
                match += (v1 == v2)
                total += 1
        return float(match) / total if total > 0 else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 7. M5: 段差分析模块
# ──────────────────────────────────────────────────────────────────────────────

@register_module
class M5_SegmentDiffModule(LogicModule):
    module_id = "M5_SegmentDiff"
    name = "段差分析"
    description = "行/列方向 diff 方差倒数"

    def analyze(self, state, cell, pv):
        r, c = cell

        # 行
        row = state._board[r, :]
        hypo_row = np.where(np.arange(state.cols)==c, pv, row)
        seq = hypo_row[~np.isnan(hypo_row)]
        score_row = self._score_seq(seq)

        # 列
        col = state._board[:, c]
        hypo_col = np.where(np.arange(state.rows)==r, pv, col)
        seq2 = hypo_col[~np.isnan(hypo_col)]
        score_col = self._score_seq(seq2)

        return max(score_row, score_col)

    @staticmethod
    def _score_seq(seq: np.ndarray) -> float:
        if seq.size < 3:
            return 0.0
        diffs = np.diff(seq)
        var = np.var(diffs)
        return 1.0 / (1.0 + var)

# ──────────────────────────────────────────────────────────────────────────────
# 8. InferenceEngine：统一加权聚合
# ──────────────────────────────────────────────────────────────────────────────

# 默认模块贡献权重，可从文件或环境加载
GLOBAL_MODULE_WEIGHTS = {
    "M1_BaseScore": 1.0,
    "M4_SymmetryAxial": 0.8,
    "M5_SegmentDiff": 0.7,
    # … 其他模块
}

class InferenceEngine:
    def __init__(self):
        self.registry = modules

    def run_inference(self, state: InternalBoardState) -> Tuple[List[ValuePrediction], List[str]]:
        results: List[ValuePrediction] = []
        warnings: List[str] = []

        # 合并权重
        weights = GLOBAL_MODULE_WEIGHTS.copy()
        if state.src.module_weights:
            weights.update(state.src.module_weights)

        for pv in state.src.proposed_values:
            if pv in state.fixed_values:
                warnings.append(f"Value {pv} 已存在于盘面，跳过。")
                results.append(ValuePrediction(proposed_value=pv, top_n_positions=[]))
                continue

            scores: List[Tuple[str,float]] = []
            for r in range(state.rows):
                for c in range(state.cols):
                    if state.is_fixed(r, c):
                        continue
                    agg = 0.0; wsum = 0.0
                    for mid, mod in self.registry.items():
                        if state.active_modules and mid not in state.active_modules:
                            continue
                        w = weights.get(mid, 1.0)
                        s = mod.analyze(state, (r, c), pv)
                        agg += s * w
                        wsum += w
                    final = agg / wsum if wsum else 0.0
                    scores.append((state.logic_code(r,c), final))

            top = sorted(scores, key=lambda x: x[1], reverse=True)[: state.src.top_n_count]
            ps = [PositionScore(position_code=code, score=round(sc,4)) for code, sc in top]
            results.append(ValuePrediction(proposed_value=pv, top_n_positions=ps))

        return results, warnings

# ──────────────────────────────────────────────────────────────────────────────
# 9. FastAPI App & 路由
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="自适应盘面推理系统", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_engine(): return InferenceEngine()

@app.post("/infer", response_model=InferenceResponse, summary="运行推理")
async def infer(board: BoardInput = Body(...), engine: InferenceEngine = Depends(get_engine)):
    start = time.perf_counter()
    try:
        state = InternalBoardState(board)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    preds, warns = await run_in_threadpool(engine.run_inference, state)
    dt = (time.perf_counter() - start) * 1000
    return InferenceResponse(predictions=preds, processing_time_ms=round(dt,2), warnings=warns or None)

@app.get("/config/modules", response_model=List[ModuleInfo], summary="可用模块列表")
def list_modules():
    return [ModuleInfo(module_id=m.module_id, name=m.name, description=m.description)
            for m in modules.values()]
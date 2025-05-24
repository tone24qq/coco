# main.py

import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator
from typing import List, Dict, Tuple, Callable
import numpy as np
from ortools.sat.python import cp_model
from celery.result import AsyncResult
from celery_worker import solve_task  # 确保 celery_worker.py 在同级

app = FastAPI(title="Plug-in權重 + 張量流 + 多強化", version="4.0")


# ── 1. 原本的向量化模組函數們（不动） ───────────────────────────
def a6_fixed_position_vec(grid: np.ndarray) -> np.ndarray:
    return grid == -1
# … 其余 MODULE_FUNCS_VEC 定义 …


MODULE_FUNCS_VEC: Dict[str, Callable] = {
    "A6": a6_fixed_position_vec,
    # … 其他 …
}
MODULE_WEIGHTS = {
    "A6": 1.0,
    # … 其他 …
}

def tensor_flow_score_vec_all(grid: np.ndarray) -> np.ndarray:
    total_score = np.zeros(grid.shape, dtype=float)
    for name, func in MODULE_FUNCS_VEC.items():
        mask = func(grid)
        total_score += mask.astype(float) * MODULE_WEIGHTS.get(name, 1.0)
    return total_score


# ── 2. 【新增】位置編碼 & 張量組成函數 ───────────────────────────
def add_positional_encoding(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    rows = np.linspace(0,1,H)[:,None] * np.ones((H,W))
    cols = np.ones((H,W)) * np.linspace(0,1,W)[None,:]
    return np.stack([rows, cols], axis=-1)

def build_feature_tensor(grid: np.ndarray) -> np.ndarray:
    """
    將 grid (H×W) 轉成 feature tensor (H×W×C)
    C = 1 + len(MODULE_FUNCS_VEC) + 2
    """
    H, W = grid.shape
    features = []
    # 1. 原始號碼通道
    features.append(grid.astype(float))
    # 2. 各延伸模組通道
    for func in MODULE_FUNCS_VEC.values():
        features.append(func(grid).astype(float))
    # 3. 位置編碼通道
    pe = add_positional_encoding(grid)
    features.append(pe[:,:,0])
    features.append(pe[:,:,1])
    # Stack 成 (H, W, C)
    return np.stack(features, axis=-1)


# ── 3. 記憶模組 & 其他輔助函數（不动） ─────────────────────────
MEM_PATH = os.path.join(os.path.dirname(__file__), "memory_cards.json")
# … 读 _memory_freq, 定义 mem_score(), get_legal_values() …


# ── 4. CP-SAT 解算函數，使用 build_feature_tensor ───────────────
def build_and_solve_cp_vec(
    grid: np.ndarray,
    candidates: List[Tuple[int,int,int]],
    legal_values
) -> List[Tuple[int,int,int,float,float]]:
    # 生成多通道特徵張量（H×W×C）
    feature_tensor = build_feature_tensor(grid)
    # 用老方法计算 base_scores
    base_scores = tensor_flow_score_vec_all(grid)
    # mem_score etc...
    # … build CP-SAT model, add constraints …
    # 將 base_scores 和 mem_score 加入目標函數
    # … solve …
    # 回傳列表 [(r,c,v,total_score, base_scores[r,c]), ...]
    return best_list  # 按你原本逻辑生成


# ── 5. API Endpoint ─────────────────────────────────────────────
class ProposedValue(BaseModel):
    pos: List[int]
    value: int

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

    @validator("new_card")
    def check_rectangular(cls, g):
        if not g or any(len(row) != len(g[0]) for row in g):
            raise ValueError("new_card 必须是矩形")
        return g

    @validator("proposed_values", each_item=True)
    def check_pv(cls, pv, values):
        grid = values.get("new_card")
        if grid:
            rows, cols = len(grid), len(grid[0])
            r, c = pv.pos
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"pos 越界：{pv.pos}")
            maxv = max(v for row in grid for v in row if v != -1)
            if pv.value < 1 or pv.value > maxv:
                raise ValueError(f"value 超出範圍：1~{maxv}")
        return pv

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    grid = np.array(req.new_card, dtype=int)
    legal = get_legal_values(grid)
    candidates = [
        (pv.pos[0], pv.pos[1], pv.value)
        for pv in req.proposed_values
        if grid[pv.pos[0], pv.pos[1]] == -1 and pv.value in legal
    ]
    if not candidates:
        raise HTTPException(400, "没有合法候选")
    best = await run_in_threadpool(build_and_solve_cp_vec, grid, candidates, legal)
    if not best:
        return {"status":"fail","result":None}
    r,c,v,score,tf_score = max(best, key=lambda x: x[3])
    return {
        "status":"success",
        "result":{
            "pos":[r,c],
            "value":v,
            "score":round(score,4),
            "tensor_flow_score":round(tf_score,4)
        }
    }
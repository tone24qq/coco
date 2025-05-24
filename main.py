import json
import os
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator
from typing import List, Dict, Tuple, Callable
import numpy as np
from ortools.sat.python import cp_model

# ── Logging Configuration ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Plug-in權重 + 張量流 + 多強化", version="4.0")

# ── 1. 原本的向量化模組函數們 ────────────────────────────────────
def a6_fixed_position_vec(grid: np.ndarray) -> np.ndarray:
    return grid == -1

def b1_row_feature_vec(grid: np.ndarray) -> np.ndarray:
    feature_map = np.zeros_like(grid, dtype=float)
    for r in range(grid.shape[0]):
        cnt = np.sum(grid[r, :] != -1)
        feature_map[r, :] = cnt
    return feature_map

def c2_col_feature_vec(grid: np.ndarray) -> np.ndarray:
    feature_map = np.zeros_like(grid, dtype=float)
    for c in range(grid.shape[1]):
        cnt = np.sum(grid[:, c] != -1)
        feature_map[:, c] = cnt
    return feature_map

MODULE_FUNCS_VEC: Dict[str, Callable] = {
    "A6": a6_fixed_position_vec,
    "B1": b1_row_feature_vec,
    "C2": c2_col_feature_vec,
}
MODULE_WEIGHTS = {
    "A6": 1.0,
    "B1": 0.5,
    "C2": 0.8,
}

def tensor_flow_score_vec_all(grid: np.ndarray) -> np.ndarray:
    total = np.zeros(grid.shape, dtype=float)
    for name, fn in MODULE_FUNCS_VEC.items():
        total += fn(grid).astype(float) * MODULE_WEIGHTS[name]
    return total

# ── 2. 增強版特徵張量 ────────────────────────────────────────────
def build_feature_tensor(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    maxv = int(np.max(grid[grid != -1])) if np.any(grid != -1) else 1
    C = 4 + maxv
    tensor = np.zeros((H, W, C), dtype=float)
    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            tensor[r, c, 0] = (v / maxv) if v != -1 else 0.0
            tensor[r, c, 1] = 1.0 if v == -1 else 0.0
            tensor[r, c, 2] = r / (H - 1) if H > 1 else 0.0
            tensor[r, c, 3] = c / (W - 1) if W > 1 else 0.0
            if v != -1:
                tensor[r, c, 4 + v - 1] = 1.0
    return tensor

def calculate_scores_from_tensor(ft: np.ndarray, grid: np.ndarray) -> np.ndarray:
    weights = np.ones(ft.shape[-1], dtype=float)
    return np.tensordot(ft, weights, axes=([2], [0]))

# ── 3. 轻量记忆模块 ────────────────────────────────────────────
MEM_PATH = os.path.join(os.path.dirname(__file__), "memory_cards.json")
_memory: Dict[str, Dict[str, float]] = {}

def _load_memory():
    global _memory
    if os.path.exists(MEM_PATH):
        try:
            _memory = json.load(open(MEM_PATH, "r", encoding="utf-8"))
            logger.info(f"Loaded memory ({len(_memory)} entries).")
        except:
            _memory = {}
    else:
        _memory = {}

_load_memory()

def _make_board_id(grid: np.ndarray) -> str:
    H, W = grid.shape
    empty = int(np.sum(grid == -1))
    return f"{H}x{W}_e{empty}"

def get_legal_values(grid: np.ndarray) -> List[int]:
    ev = grid[grid != -1]
    mv = int(np.max(ev)) if ev.size else 1
    return list(range(1, mv+1))

def mem_score(grid: np.ndarray, r: int, c: int, v: int) -> float:
    bid = _make_board_id(grid)
    key = f"{bid}_{r}_{c}_{v}"
    e = _memory.get(key)
    return (e["total_score"] / e["count"]) if e and e["count"]>0 else 0.0

def update_memory(grid: np.ndarray, r: int, c: int, v: int, score: float):
    bid = _make_board_id(grid)
    key = f"{bid}_{r}_{c}_{v}"
    ent = _memory.setdefault(key, {"count":0,"total_score":0.0})
    ent["count"] += 1
    ent["total_score"] += score

def _save_memory():
    json.dump(_memory, open(MEM_PATH, "w", encoding="utf-8"), indent=4)
    logger.info(f"Saved memory ({len(_memory)} entries).")

@app.on_event("shutdown")
def _on_shutdown():
    _save_memory()

# ── 4. CP-SAT 解算 ────────────────────────────────────────────
def build_and_solve_cp_vec(grid: np.ndarray, candidates: List[Tuple[int,int,int]], _: List[int]):
    t0 = time.time()
    ft = build_feature_tensor(grid)
    t1 = time.time()
    tf_scores = calculate_scores_from_tensor(ft, grid)
    t2 = time.time()
    logger.info(f"Tensor: {t1-t0:.3f}s, Score: {t2-t1:.3f}s")

    if not candidates:
        return []

    model = cp_model.CpModel()
    n = len(candidates)
    idx = model.NewIntVar(0, n-1, "idx")
    r_var = model.NewIntVar(0, grid.shape[0]-1, "r")
    c_var = model.NewIntVar(0, grid.shape[1]-1, "c")
    v_var = model.NewIntVar(1, int(np.max(grid[grid!=-1])) if np.any(grid!=-1) else 1, "v")

    SF = 10000
    tf_int = [int(tf_scores[r,c]*SF) for r,c,_ in candidates]
    total_int = [int((tf_scores[r,c]+mem_score(grid,r,c,v))*SF)
                 for r,c,v in candidates]

    model.AddElement(idx, tf_int, r_var)   # just to bind types
    model.AddElement(idx, [c for c,_,_ in candidates], c_var)
    model.AddElement(idx, [v for _,_,v in candidates], v_var)
    model.AddElement(idx, total_int, model.NewIntVar(min(total_int), max(total_int), "score"))

    model.Maximize(model.Objective().Copy())

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    solver.parameters.num_workers = os.cpu_count() or 1
    res = solver.Solve(model)

    out = []
    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        i = solver.Value(idx)
        r,c,v = candidates[i]
        sc = solver.Value(model.GetVarByName("score"))/SF
        tf = tf_int[i]/SF
        out = [(r,c,v,sc,tf)]
    logger.info(f"Solved in {time.time()-t0:.3f}s => {out}")
    return out

# ── 5. /analyze API ────────────────────────────────────────────
class ProposedValue(BaseModel):
    pos: List[int]
    value: int

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

    @validator("new_card")
    def _check_rect(cls, g):
        if not g or any(len(r)!=len(g[0]) for r in g):
            raise ValueError("new_card 必須矩形")
        return g

    @validator("proposed_values", each_item=True)
    def _check_pv(cls, pv, vals):
        g = vals.get("new_card")
        if g:
            rows,cols=len(g),len(g[0])
            r,c=pv.pos
            if not (0<=r<rows and 0<=c<cols): raise ValueError("pos 越界")
            mv = int(np.max(np.array(g)[np.array(g)!=-1])) if np.any(np.array(g)!=-1) else 1
            if not (1<=pv.value<=mv): raise ValueError("value 超範圍")
        return pv

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    grid = np.array(req.new_card, dtype=int)
    legal = get_legal_values(grid)

    cands=[]
    for pv in req.proposed_values:
        r,c,v = pv.pos[0],pv.pos[1],pv.value
        if grid[r,c]==-1 and v in legal:
            cands.append((r,c,v))
        else:
            logger.warning(f"跳過: {pv.pos},{pv.value}")

    if not cands:
        raise HTTPException(400,"沒有合法候選")

    best = await run_in_threadpool(build_and_solve_cp_vec, grid, cands, legal)
    if not best:
        return {"status":"fail","result":None}

    r,c,v,sc,tf = best[0]
    update_memory(grid,r,c,v,sc)
    _save_memory()

    return {
        "status":"success",
        "result":{
            "pos":[r,c],
            "value":v,
            "score":round(sc,4),
            "tensor_flow_score":round(tf,4)
        }
    }
# main.py

import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator
from typing import List, Dict, Tuple, Callable
import numpy as np
from celery.result import AsyncResult
from celery_worker import solve_task  # 請確保 celery_worker.py 在同目錄

app = FastAPI(title="Plug-in權重 + 張量流 + 自動數字範圍", version="3.1")

# 以下是 numpy 向量化規則（精簡示範）
def a6_fixed_position_vec(grid: np.ndarray) -> np.ndarray:
    return grid == -1

def m3_interval_consistency_vec_full(grid: np.ndarray) -> np.ndarray:
    R, C = grid.shape
    result = np.full((R, C), False, dtype=bool)
    for r in range(R):
        row = grid[r]
        vals = np.unique(row[row != -1])
        for val in vals:
            positions = np.where(row == val)[0]
            if len(positions) < 2:
                continue
            intervals = np.diff(positions)
            if intervals.min() <= 3:
                for pos in positions:
                    result[r, pos] = True
    return result

def a9_diagonal_symmetry_vec(grid: np.ndarray) -> np.ndarray:
    R, C = grid.shape
    result = np.zeros((R, C), dtype=bool)
    for i in range(min(R, C)):
        if grid[i, i] != -1:
            result[i, i] = True
    return result

def m5_sequence_direction_vec(grid: np.ndarray) -> np.ndarray:
    R, C = grid.shape
    result = np.zeros((R, C), dtype=bool)
    for r in range(R):
        row = grid[r]
        valid = row != -1
        vals = row[valid]
        if len(vals) > 1 and np.all(vals[:-1] < vals[1:]):
            result[r, valid] = True
    return result

def m14_mirror_diff_vec(grid: np.ndarray) -> np.ndarray:
    R, C = grid.shape
    mirror_c = C - 1 - np.arange(C)
    result = np.zeros((R, C), dtype=bool)
    for r in range(R):
        row = grid[r]
        mirrored_row = row[mirror_c]
        valid = (row != -1) & (mirrored_row != -1)
        diff = np.abs(row - mirrored_row)
        result[r, valid] = diff[valid] <= 2
    return result

def m15_parity_block_vec(grid: np.ndarray) -> np.ndarray:
    R, C = grid.shape
    pos_sum = np.add.outer(np.arange(R), np.arange(C))
    parity_pos = (pos_sum % 2 == 0)
    parity_val = (grid % 2 == 0)
    valid = (grid != -1)
    return valid & (parity_val == parity_pos)

MODULE_FUNCS_VEC: Dict[str, Callable] = {
    "A6": a6_fixed_position_vec,
    "M3": m3_interval_consistency_vec_full,
    "A9": a9_diagonal_symmetry_vec,
    "M5": m5_sequence_direction_vec,
    "M14": m14_mirror_diff_vec,
    "M15": m15_parity_block_vec,
}

MODULE_WEIGHTS = {
    "A6": 1.0, "M3": 1.2, "A9": 1.0, "M5": 1.1,
    "M14": 1.0, "M15": 1.1,
}

def tensor_flow_score_vec_all(grid: np.ndarray) -> np.ndarray:
    total_score = np.zeros(grid.shape, dtype=float)
    for name, func in MODULE_FUNCS_VEC.items():
        try:
            mask = func(grid)
            weight = MODULE_WEIGHTS.get(name, 1.0)
            total_score += mask.astype(float) * weight
        except Exception:
            pass
    return total_score

def get_card_max_value(grid):
    return max(v for row in grid for v in row if v != -1)

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
            card_max = get_card_max_value(grid)
            if pv.value < 1 or pv.value > card_max:
                raise ValueError(f"value 超出卡片最大值範圍：1~{card_max}")
        return pv

MEM_PATH = os.path.join(os.path.dirname(__file__), "memory_cards.json")
_memory_freq: Dict[Tuple[int,int,int], int] = {}
_total_samples = 0
if os.path.exists(MEM_PATH):
    data = json.load(open(MEM_PATH, "r", encoding="utf-8"))
    for card in data.get("memory_cards", []):
        for r, row in enumerate(card):
            for c, v in enumerate(row):
                if v != -1:
                    _memory_freq[(r, c, v)] = _memory_freq.get((r, c, v), 0) + 1
                    _total_samples += 1

def mem_score(r, c, v, legal_values):
    if v not in legal_values:
        return 0.0
    cnt = _memory_freq.get((r, c, v), 0)
    return cnt / _total_samples if _total_samples else 0.0

def get_legal_values(grid: np.ndarray):
    card_max = get_card_max_value(grid)
    used = set(grid.flatten()[grid.flatten() != -1])
    return set(v for v in range(1, card_max+1) if v not in used)

def build_and_solve_cp_vec(grid: np.ndarray, candidates: List[Tuple[int,int,int]], legal_values):
    model = cp_model.CpModel()
    rows, cols = grid.shape
    used = set(grid.flatten()[grid.flatten() != -1])
    x = [model.NewBoolVar(f"x_{i}") for i in range(len(candidates))]
    model.Add(sum(x) == 1)

    for i, (r, c, v) in enumerate(candidates):
        if v in used or v not in legal_values:
            model.Add(x[i] == 0)
    for i in range(len(candidates)):
        vi = candidates[i][2]
        for j in range(i+1, len(candidates)):
            vj = candidates[j][2]
            if vi == vj:
                model.Add(x[i] + x[j] <= 1)

    scores = tensor_flow_score_vec_all(grid)
    weights = []
    tensor_scores = []

    for i, (r, c, v) in enumerate(candidates):
        score = scores[r, c]
        score += 5.0 * mem_score(r, c, v, legal_values)
        tensor_scores.append(scores[r, c])
        weights.append(int(score * 1000))

    obj = sum(x[i] * weights[i] for i in range(len(candidates)))
    model.Maximize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5
    solver.parameters.num_search_workers = os.cpu_count() or 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []

    best = []
    for i in range(len(candidates)):
        if solver.Value(x[i]):
            best.append((candidates[i][0], candidates[i][1], candidates[i][2], weights[i] / 1000.0, tensor_scores[i]))
    return best

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        grid = np.array(req.new_card, dtype=int)
        legal_values = get_legal_values(grid)
        candidates = []
        for pv in req.proposed_values:
            r, c, v = pv.pos[0], pv.pos[1], pv.value
            if grid[r, c] != -1:
                continue
            if v in legal_values:
                candidates.append((r, c, v))
        if not candidates:
            raise HTTPException(400, "没有合法可选候选")
        best = await run_in_threadpool(build_and_solve_cp_vec, grid, candidates, legal_values)
        if best:
            r, c, v, s, tensor = max(best, key=lambda x: x[3])
            return {
                "status": "success",
                "result": {
                    "pos": [r, c],
                    "value": v,
                    "score": round(s, 4),
                    "tensor_flow_score": round(tensor, 4),
                    "plugin_detail": None,
                    "tensor_flow_detail": None,
                    "mem_score": None,
                    "total_score": s
                },
                "all_candidates_detail": []
            }
        else:
            return {"status": "fail", "result": None, "all_candidates_detail": []}
    except Exception as e:
        return {"status": "error", "message": str(e)}
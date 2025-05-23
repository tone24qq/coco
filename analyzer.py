# analyzer.py
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from ortools.sat.python import cp_model

app = FastAPI(title="极限补格AI分析器（CP-SAT 版防呆加強）", version="2.1")

# —— 1. 输入/输出 数据模型 ——#
class ProposedValue(BaseModel):
    pos: List[int]      # [row, col]
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
            N = rows * cols
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"pos 越界：{pv.pos}")
            if pv.value < 1 or pv.value > N:
                raise ValueError(f"value 超出合法范围：1~{N}")
        return pv

# —— 2. 记忆样本（M0）加载 ——#
@dataclass(frozen=True)
class EvalContext:
    grid: np.ndarray
    rows: int
    cols: int

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
    # 只對當前卡片合法號碼記憶共鳴
    if v not in legal_values:
        return 0.0
    cnt = _memory_freq.get((r, c, v), 0)
    return cnt / _total_samples if _total_samples else 0.0

def get_legal_values(grid):
    N = grid.shape[0] * grid.shape[1]
    used = set(grid.flatten()[grid.flatten() != -1])
    return [v for v in range(1, N+1) if v not in used]

# —— 3. 约束 & 打分函数 ——#
def build_and_solve_cp(grid: np.ndarray, candidates: List[Tuple[int,int,int]], legal_values):
    model = cp_model.CpModel()
    rows, cols = grid.shape
    N = rows * cols

    # 已经填好的数字集合
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

    weights = []
    for i, (r, c, v) in enumerate(candidates):
        score = 0.0
        score += 5.0 * mem_score(r, c, v, legal_values)
        if r in (0, rows-1) or c in (0, cols-1):
            score += 0.5
        seq = grid[r][grid[r] != -1]
        if seq.size == 0 or np.all(seq[:-1] < seq[1:]):
            score += 0.3
        if r < rows // 2:
            score += 0.2
        cent_r, cent_c = (rows-1)/2, (cols-1)/2
        if abs(r-cent_r) == abs(c-cent_c):
            score += 0.2
        dist_matches = np.where(grid[r] == v)[0]
        if dist_matches.size:
            d = np.min(np.abs(dist_matches - c))
            score += 0.4 / (1 + d)
        weights.append(int(score * 1000))

    obj = sum(x[i] * weights[i] for i in range(len(candidates)))
    model.Maximize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []

    best = []
    for i in range(len(candidates)):
        if solver.Value(x[i]):
            best.append((candidates[i][0], candidates[i][1], candidates[i][2], weights[i] / 1000.0))
    return best

# —— 4. API 逻辑 ——#
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    grid = np.array(req.new_card, dtype=int)
    rows, cols = grid.shape
    legal_values = set(get_legal_values(grid))

    candidates = []
    for pv in req.proposed_values:
        r, c, v = pv.pos[0], pv.pos[1], pv.value
        if grid[r, c] != -1:
            continue
        # 嚴格只允許合法號碼
        if v in legal_values:
            candidates.append((r, c, v))
    if not candidates:
        raise HTTPException(400, "没有合法可选候选")

    best = build_and_solve_cp(grid, candidates, legal_values)
    best = sorted(best, key=lambda x: -x[3])[:3]

    return {
        "status": "success",
        "results": [
            {"pos": [r, c], "value": v, "score": round(s, 4)}
            for r, c, v, s in best
        ]
    }

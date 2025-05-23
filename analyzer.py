# analyzer.py
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

# **新增**：OR-Tools CP-SAT
from ortools.sat.python import cp_model

app = FastAPI(title="极限补格AI分析器（CP-SAT 版）", version="2.0")

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
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"pos 越界：{pv.pos}")
            if pv.value < 1 or pv.value > rows*cols:
                raise ValueError(f"value 超出合法范围：1~{rows*cols}")
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
                    _memory_freq[(r,c,v)] = _memory_freq.get((r,c,v),0) + 1
                    _total_samples += 1

def mem_score(r,c,v):
    cnt = _memory_freq.get((r,c,v), 0)
    return cnt / _total_samples if _total_samples else 0.0

# —— 3. 约束 & 打分函数 ——#
def build_and_solve_cp(grid: np.ndarray, candidates: List[Tuple[int,int,int]]):
    """
    构建 CP-SAT 模型：
    - 变量 x[i] = True 表示选中 candidates[i]
    - 每个位置只能选一个值
    - 保证所有已定值 + 新选的值互不重复
    - 尽量最大化记忆频率 + 位置规则打分
    """
    model = cp_model.CpModel()
    rows, cols = grid.shape
    N = rows * cols

    # 已经填好的数字集合
    used = set(grid.flatten()[grid.flatten() != -1])

    # Bool 变量
    x = [model.NewBoolVar(f"x_{i}") for i in range(len(candidates))]

    # 1) 每个候选 (r,c,v) 只能选一次
    #    但我们要选 TOP K，比如选最多1个（你也可以改成选3个一起返回）
    model.Add(sum(x) == 1)

    # 2) 唯一性约束：选中的值不能跟 used 冲突，也不能彼此冲突
    #    例如若有 (r1,c1,v1) 和 (r2,c2,v2) ，v1≠v2
    for i,(r,c,v) in enumerate(candidates):
        # 值已存在则禁止
        if v in used:
            model.Add(x[i] == 0)
    # 互斥（值重复）
    for i in range(len(candidates)):
        ri,ci,vi = candidates[i]
        for j in range(i+1, len(candidates)):
            rj,cj,vj = candidates[j]
            if vi == vj:
                model.Add(x[i] + x[j] <= 1)

    # 3) 构建目标：记忆打分 + 位置启发式
    #    记忆频率 M0, 越靠近中心加分 M13, 边缘加分 M10, 行递增 M5……
    weights = []
    for i,(r,c,v) in enumerate(candidates):
        score = 0.0
        # M0
        score += 5.0 * mem_score(r,c,v)
        # M10 边缘
        if r in (0,rows-1) or c in (0,cols-1):
            score += 0.5
        # M5 行递增
        seq = grid[r][grid[r]!=-1]
        if seq.size==0 or np.all(seq[:-1] < seq[1:]):
            score += 0.3
        # M16 偏上
        if r < rows//2:
            score += 0.2
        # M13 中心对角
        cent_r, cent_c = (rows-1)/2, (cols-1)/2
        if abs(r-cent_r)==abs(c-cent_c):
            score += 0.2
        # M3 同行距离
        dist_matches = np.where(grid[r]==v)[0]
        if dist_matches.size:
            d = np.min(np.abs(dist_matches - c))
            score += 0.4/(1+d)
        # ……你可以在这里任意加
        # 最后乘上一个放大系数
        weights.append(int(score * 1000))

    obj = sum(x[i] * weights[i] for i in range(len(candidates)))
    model.Maximize(obj)

    # 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5  # 手机版要限速
    solver.parameters.num_search_workers = 4     # 并行
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []

    # 返回选中的候选
    best = []
    for i in range(len(candidates)):
        if solver.Value(x[i]):
            best.append((candidates[i][0], candidates[i][1], candidates[i][2],
                         weights[i]/1000.0))
    return best

# —— 4. API 逻辑 ——#
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    grid = np.array(req.new_card, dtype=int)
    rows, cols = grid.shape

    # 收集所有可行的 (r,c,v)
    candidates = []
    used = set(grid.flatten()[grid.flatten()!=-1])
    for pv in req.proposed_values:
        r,c,v = pv.pos[0], pv.pos[1], pv.value
        if grid[r,c] != -1:
            continue
        if 1 <= v <= rows*cols:
            candidates.append((r,c,v))
    if not candidates:
        raise HTTPException(400, "没有可选候选")

    best = build_and_solve_cp(grid, candidates)
    # 返回 top3
    best = sorted(best, key=lambda x: -x[3])[:3]

    return {
        "status": "success",
        "results": [
            {"pos":[r,c], "value":v, "score":round(s,4)}
            for r,c,v,s in best
        ]
    }
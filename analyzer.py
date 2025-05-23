import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Dict, Tuple, Callable
import numpy as np
from ortools.sat.python import cp_model

app = FastAPI(title="Plug-in權重 + 張量流 + 自動數字範圍", version="3.1")

# —— Plug-in規則 ——#
def a6_fixed_position(grid, pos, value):      return grid[pos[0]][pos[1]] == -1
def m3_interval_consistency(grid, pos, value):
    r, c = pos
    ints = [abs(c - j) for j, v in enumerate(grid[r]) if v == value]
    return ints[0] if ints else -1
def a9_diagonal_symmetry(grid, pos, value):    return pos[0] == pos[1] and grid[pos[0]][pos[1]] == value
def m5_sequence_direction(grid, pos, value):
    row = [v for v in grid[pos[0]] if v != -1]
    return all(x < y for x, y in zip(row, row[1:]))
def m10_edge_prediction(grid, pos, value):
    r, c = pos; R, C = len(grid), len(grid[0])
    return r in (0, R-1) or c in (0, C-1)
def m11_jump_zone(grid, pos, value):          return (pos[0] + pos[1]) % 2 == 0
def m12_diagonal_repeat(grid, pos, value):    return a9_diagonal_symmetry(grid, pos, value)
def m13_center_rotation(grid, pos, value):
    R, C = len(grid), len(grid[0]); cr, cc = R//2, C//2
    return abs(pos[0]-cr) == abs(pos[1]-cc)
def m14_mirror_diff(grid, pos, value):
    r, c = pos; C = len(grid[0]); mc = C-1-c
    if 0 <= mc < C and grid[r][mc] != -1:
        return abs(grid[r][c] - grid[r][mc]) <= 2
    return False
def m15_parity_block(grid, pos, value):       return (value % 2 == 0) == ((pos[0] + pos[1]) % 2 == 0)
def m16_upper_lower_ratio(grid, pos, value):  return pos[0] < (len(grid)//2)
def m17_column_mirror_reverse(grid, pos, value):
    C = len(grid[0]); r, c = pos
    return grid[r][c] == grid[r][C-1-c]
def m18_horizontal_delta(grid, pos, value):
    r, c = pos; C = len(grid[0])
    if 0 < c < C-1 and grid[r][c-1] != -1 and grid[r][c+1] != -1:
        return abs(grid[r][c-1] - grid[r][c+1]) <= 2
    return False
def m19_arc_shape(grid, pos, value):          return abs(pos[0] - pos[1]) <= 1

MODULE_FUNCS: Dict[str, Callable] = {
    "A6": a6_fixed_position,
    "M3": m3_interval_consistency,
    "A9": a9_diagonal_symmetry,
    "M5": m5_sequence_direction,
    "M10": m10_edge_prediction,
    "M11": m11_jump_zone,
    "M12": m12_diagonal_repeat,
    "M13": m13_center_rotation,
    "M14": m14_mirror_diff,
    "M15": m15_parity_block,
    "M16": m16_upper_lower_ratio,
    "M17": m17_column_mirror_reverse,
    "M18": m18_horizontal_delta,
    "M19": m19_arc_shape,
}
MODULE_WEIGHTS = {
    "A6": 1.0, "M3": 1.2, "A9": 1.0, "M5": 1.1,
    "M10": 0.9, "M11": 0.8, "M12": 1.0, "M13": 1.0,
    "M14": 1.0, "M15": 1.1, "M16": 0.7, "M17": 1.0,
    "M18": 0.9, "M19": 0.8,
}

def tensor_flow_score(grid, pos, value):
    new_grid = [row[:] for row in grid]  # 補格後新張量
    new_grid[pos[0]][pos[1]] = value
    score = 0.0
    for r in range(len(new_grid)):
        for c in range(len(new_grid[0])):
            if new_grid[r][c] == -1:
                continue
            for name, func in MODULE_FUNCS.items():
                try:
                    if func(new_grid, (r, c), new_grid[r][c]):
                        score += MODULE_WEIGHTS.get(name, 1.0)
                except Exception:
                    pass
    return score

# ——— 自動調適最大值 ——— #
def get_card_max_value(grid):
    return max(v for row in grid for v in row if v != -1)

# —— API模型 —— #
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
            N = rows * cols
            r, c = pv.pos
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"pos 越界：{pv.pos}")
            # 嚴格只收小於等於卡片最大值的數字
            card_max = get_card_max_value(grid)
            if pv.value < 1 or pv.value > card_max:
                raise ValueError(f"value 超出卡片最大值範圍：1~{card_max}")
        return pv

# 記憶體分數（照舊）
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

def get_legal_values(grid):
    card_max = get_card_max_value(grid)
    N = grid.shape[0] * grid.shape[1]
    used = set(grid.flatten()[grid.flatten() != -1])
    # 嚴格限制：1~card_max 並未被用過
    return set(v for v in range(1, card_max+1) if v not in used)

# —— Build & Solve (Plug-in + 張量流 + 自動範圍) ——#
def build_and_solve_cp(grid: np.ndarray, candidates: List[Tuple[int,int,int]], legal_values):
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
        for j in range(i+1, len(c
        def build_and_solve_cp(grid: np.ndarray, candidates: List[Tuple[int,int,int]], legal_values):
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

    weights = []
    tensor_scores = []
    for i, (r, c, v) in enumerate(candidates):
        score = 0.0
        # Plug-in模組加權
        for name, func in MODULE_FUNCS.items():
            try:
                if func(grid, (r, c), v):
                    score += MODULE_WEIGHTS.get(name, 1.0)
            except Exception:
                pass
        # 張量流分數
        tensor_score = tensor_flow_score(grid, (r, c), v)
        tensor_scores.append(tensor_score)
        score += tensor_score * 1.0
        # 記憶體共鳴分數
        score += 5.0 * mem_score(r, c, v, legal_values)
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
            best.append((candidates[i][0], candidates[i][1], candidates[i][2], weights[i] / 1000.0, tensor_scores[i]))
    return best

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        # ======= 這裡開始全部主體邏輯 =======
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
        best, detail_log = build_and_solve_cp(grid, candidates, legal_values)
        if best:
            r, c, v, s, tensor = max(best, key=lambda x: x[3])
            detail_item = next(item for item in detail_log if item["pos"] == [r, c] and item["value"] == v)
            return {
                "status": "success",
                "result": {
                    "pos": [r, c],
                    "value": v,
                    "score": round(s, 4),
                    "tensor_flow_score": round(tensor, 4),
                    "plugin_detail": detail_item["plugin_detail"],
                    "tensor_flow_detail": detail_item["tensor_flow_detail"],
                    "mem_score": detail_item["mem_score"],
                    "total_score": detail_item["total_score"]
                },
                "all_candidates_detail": detail_log
            }
        else:
            return {"status": "fail", "result": None, "all_candidates_detail": detail_log}
        # ======= 這裡結束全部主體邏輯 =======
    except Exception as e:
        # 這裡會直接把詳細錯誤原因顯示給你
        return {
            "status": "error",
            "message": str(e)
        }
: "fail", "result": None}
    except Exception as e:
        # 這裡會直接把詳細錯誤原因顯示給你
        return {"status": "error", "message": str(e)}

        }
    else:
        return {
            "status": "fail",
            "result": None
        }


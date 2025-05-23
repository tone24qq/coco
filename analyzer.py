from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Callable

app = FastAPI()

# ── Models ────────────────────────────────────────────────────────────────────

class ProposedValue(BaseModel):
    pos: List[int]      # [row, col]
    value: int
    score: float
    py_guide: str

class SimplifiedAnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]
    weight_base: float = 0.7   # 原始 API 分数权重
    weight_meta: float = 0.3   # 死逻辑模块打分权重

# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_grid(grid: List[List[int]]) -> str:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    col_labels = "   " + "".join(f" C{c+1:<3}" for c in range(cols)) + "\n"
    sep = "  +" + "+".join("----" for _ in range(cols)) + "+\n"
    body = ""
    for r, row in enumerate(grid):
        vals = "|".join(f"{str(v) if v != -1 else '':>3} " for v in row)
        body += f"R{r+1:<2}|{vals}|\n" + sep
    return "```lua\n" + col_labels + sep + body + "```"

# ── Dead‐logic Modules ────────────────────────────────────────────────────────

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

# 模块工厂映射
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

# ── 权重 & 二阶模块 ────────────────────────────────────────────────────────────

MODULE_WEIGHTS = {
    "A6": 1.0, "M3": 1.2, "A9": 1.0, "M5": 1.1,
    "M10": 0.9, "M11": 0.8, "M12": 1.0, "M13": 1.0,
    "M14": 1.0, "M15": 1.1, "M16": 0.7, "M17": 1.0,
    "M18": 0.9, "M19": 0.8,
}

SECONDARY_MODULES: Dict[str, Callable[[Dict[str,bool]], bool]] = {
    "M3∧M5": lambda mr: mr["M3"] and mr["M5"],
    "M10⊕M11": lambda mr: mr["M10"] ^ mr["M11"],
}

def aggregate_module_score(mr: Dict[str,bool]) -> float:
    return sum(MODULE_WEIGHTS.get(name, 1.0) * (1.0 if passed else 0.0)
               for name, passed in mr.items())

def aggregate_with_secondary(mr: Dict[str,bool]) -> float:
    score = aggregate_module_score(mr)
    for extra, fn in SECONDARY_MODULES.items():
        if fn(mr):
            score += 0.5
    return score

# ── 分析 Endpoint ────────────────────────────────────────────────────────────

@app.post("/analyze")
def analyze(req: SimplifiedAnalyzeRequest):
    # 基本校验
    if not req.new_card or not req.proposed_values:
        return {"status": "error", "message": "缺少必要字段"}
    if any(len(row) != len(req.new_card[0]) for row in req.new_card):
        return {"status": "error", "message": "表格列长度不一致"}

    items = []
    for pv in req.proposed_values:
        # 1) 模块结果
        mod_results = {
            name: func(req.new_card, pv.pos, pv.value)
            for name, func in MODULE_FUNCS.items()
        }
        # 2) 计算 meta_score
        meta = aggregate_with_secondary(mod_results)
        # 3) 融合原始 score 与 meta_score
        combined = req.weight_base * pv.score + req.weight_meta * meta

        items.append({
            "pv": pv,
            "modules": mod_results,
            "meta_score": round(meta, 4),
            "combined_score": round(combined, 4),
        })

    # 排序取 Top3
    top3 = sorted(items, key=lambda x: -x["combined_score"])[:3]
    visual = visualize_grid(req.new_card)

    # 组织返回
    results = []
    for it in top3:
        pv = it["pv"]
        results.append({
            "pos": pv.pos,
            "value": pv.value,
            "base_score": round(pv.score, 4),
            "meta_score": it["meta_score"],
            "combined_score": it["combined_score"],
            "modules": it["modules"],
            "py_guide": pv.py_guide
        })

    return {
        "status": "success",
        "visual_grid": visual,
        "results": results
    }
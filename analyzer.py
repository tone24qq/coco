from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# ─── Data models ───────────────────────────────────────────────────────────────
class ProposedValue(BaseModel):
    pos: List[int]       # [row, col]
    value: int           # 填入的数字
    score: float = 0.0   # 规则累计得分
    py_guide: str = ""   # 命中规则标签

class SimplifiedAnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

@dataclass
class EvalContext:
    grid: np.ndarray
    r: int
    c: int
    v: int
    rows: int
    cols: int
    max_val: int
    row_valid: set
    col_valid: set
    val_range: Tuple[int,int]
    used_values: set

# ─── 规则实现 ────────────────────────────────────────────────────────────────
def A6(ctx: EvalContext) -> Tuple[float,str]:
    """A6: 固定空位"""
    return (1.0, "A6") if ctx.grid[ctx.r, ctx.c] == -1 else (0.0, "")

def M3(ctx: EvalContext) -> Tuple[float,str]:
    """M3: 同行已有时距一致性"""
    row = ctx.grid[ctx.r]
    idx = np.where(row == ctx.v)[0]
    if idx.size:
        d = abs(ctx.c - idx[0])
        score = max(0.0, 1.0 - d / ctx.cols)
        return (score, f"M3(d={d})")
    return (0.0, "")

def M5_row_increasing(ctx: EvalContext) -> Tuple[float,str]:
    """M5_row_increasing: 本行剩余数字递增序列"""
    vals = ctx.grid[ctx.r][ctx.grid[ctx.r] != -1]
    if vals.size > 1 and np.all(np.diff(vals) > 0):
        return (1.0, "M5_row_increasing")
    return (0.0, "")

def M10_edge(ctx: EvalContext) -> Tuple[float,str]:
    """M10: 边缘优先"""
    if ctx.r in (0, ctx.rows-1) or ctx.c in (0, ctx.cols-1):
        return (1.0, "M10_edge")
    return (0.0, "")

def M13_center_diag(ctx: EvalContext) -> Tuple[float,str]:
    """M13: 对角线从中心"""
    center = (ctx.rows//2, ctx.cols//2)
    if abs(ctx.r - center[0]) == abs(ctx.c - center[1]):
        return (1.0, "M13_center_diag")
    return (0.0, "")

def M14_mirror_diff(ctx: EvalContext) -> Tuple[float,str]:
    """M14: 镜像对称差距 ≤ 2"""
    mc = ctx.cols - 1 - ctx.c
    if 0 <= mc < ctx.cols:
        a, b = ctx.grid[ctx.r, ctx.c], ctx.grid[ctx.r, mc]
        if a != -1 and b != -1 and abs(a - b) <= 2:
            return (1.0, "M14_mirror_diff")
    return (0.0, "")

def M15_parity(ctx: EvalContext) -> Tuple[float,str]:
    """M15: 偶奇区块"""
    if (ctx.v % 2 == (ctx.r + ctx.c) % 2):
        return (1.0, "M15_parity")
    return (0.0, "")

# ─── 规则注册 ────────────────────────────────────────────────────────────────
RULES = {
    "A6": A6,
    "M3": M3,
    "M5_row_increasing": M5_row_increasing,
    "M10_edge": M10_edge,
    "M13_center_diag": M13_center_diag,
    "M14_mirror_diff": M14_mirror_diff,
    "M15_parity": M15_parity,
}

# ─── 表格可视化 ─────────────────────────────────────────────────────────────
def visualize_grid(grid: List[List[int]]) -> str:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    header = "   " + "".join(f" C{c+1:<3}" for c in range(cols)) + "\n"
    sep = "  +" + "+".join("----" for _ in range(cols)) + "+\n"
    body = ""
    for r, row in enumerate(grid):
        vals = "|".join(f"{str(v) if v != -1 else '':>3} " for v in row)
        body += f"R{r+1:<2}|{vals}|\n" + sep
    return "```lua\n" + header + sep + body + "```"

# ─── 主接口 ─────────────────────────────────────────────────────────────────
@app.post("/analyze")
def analyze(req: SimplifiedAnalyzeRequest):
    # 转 numpy 数组 & 基本校验
    try:
        grid = np.array(req.new_card, dtype=int)
    except Exception:
        raise HTTPException(400, "new_card 必须为二维整数列表")
    if grid.ndim != 2:
        raise HTTPException(400, "new_card 必须为二维结构")
    rows, cols = grid.shape
    max_val = rows * cols
    if any(len(row) != cols for row in req.new_card):
        raise HTTPException(400, "表格列长度不一致")

    # 预计算 cache
    row_valid = set(range(rows))
    col_valid = set(range(cols))
    val_range = (1, max_val)
    used_values = set(grid[grid != -1].flatten())

    # 单个候选评估
    def evaluate(pv: ProposedValue) -> ProposedValue:
        r, c = pv.pos
        v = pv.value
        # 越界 & 值域 & 重复验证
        if r not in row_valid or c not in col_valid:
            raise ValueError(f"pos 越界: {pv.pos}")
        if not (val_range[0] <= v <= val_range[1]):
            raise ValueError(f"value 必须在1~{max_val}之间: {v}")
        if v in used_values:
            raise ValueError(f"value 已存在: {v}")
        ctx = EvalContext(grid, r, c, v, rows, cols, max_val,
                          row_valid, col_valid, val_range, used_values)
        total = 0.0
        tags = []
        for fn in RULES.values():
            inc, tag = fn(ctx)
            if inc > 0:
                total += inc
                tags.append(tag)
        pv.score = total
        pv.py_guide = ",".join(tags)
        return pv

    # 并发评估
    try:
        with ThreadPoolExecutor() as executor:
            evaluated = list(executor.map(evaluate, req.proposed_values))
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 排序并取 Top3
    top3 = sorted(evaluated, key=lambda x: -x.score)[:3]

    return {
        "status": "success",
        "visual_grid": visualize_grid(req.new_card),
        "results": [
            {"pos": p.pos, "value": p.value, "score": p.score, "py_guide": p.py_guide}
            for p in top3
        ]
    }
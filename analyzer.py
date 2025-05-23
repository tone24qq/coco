from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np

app = FastAPI()

class ProposedValue(BaseModel):
    pos: List[int]       # [row, col]
    value: int           # 要填的数字
    score: float = 0.0   # 规则评分
    py_guide: str = ""   # 规则名称和参数

class SimplifiedAnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

# ─── 模组规则实现 ───────────────────────────────────────────────────────────
def A6(g: np.ndarray, r: int, c: int, v: int) -> Tuple[float,str]:
    """固定空位"""
    return (1.0, "A6") if g[r, c] == -1 else (0.0, "")

def M3(g: np.ndarray, r: int, c: int, v: int) -> Tuple[float,str]:
    """同行已有时距一致性，越近越高"""
    row = g[r]
    idx = np.where(row == v)[0]
    if idx.size:
        d = abs(c - idx[0])
        score = max(0.0, 1 - d / g.shape[1])
        return (score, f"M3(d={d})")
    return (0.0, "")

def M5(g: np.ndarray, r: int, c: int, v: int) -> Tuple[float,str]:
    """同列/同排数字递增序列"""
    row = g[r][g[r] != -1]
    if row.size > 1 and np.all(np.diff(row) > 0):
        return (1.0, "M5")
    return (0.0, "")

# …可继续添加其他规则…
RULES = {
    "A6": A6,
    "M3": M3,
    "M5": M5,
    # "A9": A9, "M10": M10, …
}

# ─── 视图化表格 ───────────────────────────────────────────────────────────
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

# ─── 主分析接口 ───────────────────────────────────────────────────────────
@app.post("/analyze")
def analyze(req: SimplifiedAnalyzeRequest):
    # 转成 numpy 数组
    try:
        grid = np.array(req.new_card, dtype=int)
    except Exception:
        raise HTTPException(status_code=400, detail="new_card 必须是二维整数列表")
    if grid.ndim != 2:
        raise HTTPException(status_code=400, detail="new_card 必须是二维结构")
    rows, cols = grid.shape
    max_val = rows * cols

    # 验证每行长度一致
    if any(len(row) != cols for row in req.new_card):
        raise HTTPException(status_code=400, detail="表格列长度不一致")

    evaluated: List[ProposedValue] = []
    for pv in req.proposed_values:
        r, c = pv.pos
        v = pv.value
        # 越界 & 数值范围检测
        if not (0 <= r < rows and 0 <= c < cols):
            raise HTTPException(status_code=400, detail=f"pos 越界: {pv.pos}")
        if not (1 <= v <= max_val):
            raise HTTPException(status_code=400, detail=f"value 必须在1~{max_val}之间: {v}")

        # 执行所有规则累加评分
        total, tags = 0.0, []
        for name, fn in RULES.items():
            inc, tag = fn(grid, r, c, v)
            if inc > 0:
                total += inc
                if tag:
                    tags.append(tag)
        pv.score = total
        pv.py_guide = ",".join(tags)
        evaluated.append(pv)

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
# analyzer.py
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
import numpy as np
import os

app = FastAPI(title="极限补格AI分析器", version="1.0")

# —— 1. 数据模型 ——#
class ProposedValue(BaseModel):
    pos: List[int]      # [row, col]
    value: int
    score: float = 0.0
    py_guide: str = ""

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

# —— 2. EvalContext & 记忆样本 ——#
@dataclass(frozen=True)
class EvalContext:
    grid: np.ndarray
    row: int
    col: int
    val: int
    rows: int
    cols: int
    used: set

# 记忆样本权重 (M0)
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

def M0_memory_freq(ctx: EvalContext) -> Tuple[float,str]:
    cnt = _memory_freq.get((ctx.row,ctx.col,ctx.val),0)
    score = cnt / _total_samples if _total_samples else 0.0
    return score, f"M0_mem({cnt}/{_total_samples})"

# —— 3. 规则辞典 ——#
def A6(ctx):  # 空格才考虑
    return (1.0, "A6_empty") if ctx.grid[ctx.row,ctx.col] == -1 else (0.0, "")

def M3(ctx):  # 同行若已有相同值，则距离越近分越高
    cols = ctx.cols
    row = ctx.grid[ctx.row]
    matches = np.where(row == ctx.val)[0]
    if matches.size:
        d = np.min(np.abs(matches - ctx.col))
        return (1.0/(1+d), f"M3_d{d}")
    return (0.0, "")

def A9(ctx):  # 主对角线对称
    return (1.0, "A9_diag") if ctx.row == ctx.col and ctx.val == ctx.grid[ctx.row,ctx.col] else (0.0, "")

def M5_row_inc(ctx):
    row = ctx.grid[ctx.row]
    seq = row[row!=-1]
    if seq.size == 0 or np.all(seq[:-1] < seq[1:]):
        return (1.0, "M5_row_inc")
    return (0.0, "")

def M10_edge(ctx):
    r,c = ctx.row, ctx.col
    return (1.0, "M10_edge") if (r in (0,ctx.rows-1) or c in (0,ctx.cols-1)) else (0.0, "")

def M11_jump(ctx):
    return (1.0, "M11_jump") if ((ctx.row+ctx.col)%2==0) == (ctx.val%2==0) else (0.0, "")

def M12_diag_rep(ctx):
    return (1.0, "M12_diag_rep") if (ctx.row==ctx.col and ctx.grid[ctx.row,ctx.col]==ctx.val) else (0.0, "")

def M13_ctr_rot(ctx):
    cent = ((ctx.rows-1)/2, (ctx.cols-1)/2)
    return (1.0, "M13_ctr_rot") if abs(ctx.row-cent[0])==abs(ctx.col-cent[1]) else (0.0, "")

def M14_mirror(ctx):
    mc = ctx.cols-1-ctx.col
    if ctx.grid[ctx.row,mc] != -1:
        return ((2-abs(ctx.val-ctx.grid[ctx.row,mc]))/2, f"M14_mirror({ctx.grid[ctx.row,mc]})") if abs(ctx.val-ctx.grid[ctx.row,mc])<=2 else (0.0,"")
    return (0.0,"")

def M15_parity(ctx):
    return (1.0, "M15_parity") if (ctx.val%2)==((ctx.row+ctx.col)%2) else (0.0,"")

def M16_uratio(ctx):
    return (1.0, "M16_upper") if ctx.row < ctx.rows//2 else (0.0,"")

def M17_col_mir(ctx):
    mc = ctx.cols-1-ctx.col
    return (1.0, "M17_col_mir") if ctx.grid[ctx.row,mc]==ctx.val else (0.0,"")

def M18_hdelta(ctx):
    r,c=ctx.row,ctx.col
    left = ctx.grid[r,c-1] if c>0 else -1
    right= ctx.grid[r,c+1] if c<ctx.cols-1 else -1
    if left!=-1 and right!=-1:
        return ((2-abs(left-right))/2, f"M18_hdelta({left},{right})") if abs(left-right)<=2 else (0.0,"")
    return (0.0,"")

def M19_arc(ctx):
    return (1.0, "M19_arc") if abs(ctx.row-ctx.col)<=1 else (0.0,"")

RULES: Dict[str, Callable[[EvalContext], Tuple[float,str]]] = {
    "M0": M0_memory_freq,
    "A6": A6, "M3": M3, "A9": A9,
    "M5": M5_row_inc, "M10": M10_edge, "M11": M11_jump,
    "M12": M12_diag_rep, "M13": M13_ctr_rot, "M14": M14_mirror,
    "M15": M15_parity, "M16": M16_uratio, "M17": M17_col_mir,
    "M18": M18_hdelta, "M19": M19_arc,
}

# —— 4. 分析逻辑 ——#
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    grid = np.array(req.new_card, dtype=int)
    rows, cols = grid.shape
    used = set(grid.flatten()[grid.flatten()!=-1])

    results = []
    for pv in req.proposed_values:
        ctx = EvalContext(grid=grid, row=pv.pos[0], col=pv.pos[1],
                          val=pv.value, rows=rows, cols=cols, used=used)
        total_score = 0.0
        guides = []
        # 检查不重复
        if pv.value in used:
            pv.score = 0.0
            pv.py_guide = "INVALID_DUP"
            results.append(pv)
            continue
        # 逐规则打分
        for name, fn in RULES.items():
            sc, g = fn(ctx)
            if sc:
                total_score += sc
                guides.append(name+":"+g)
        # 动态归一
        denom = len(RULES)
        pv.score = total_score / denom
        pv.py_guide = ",".join(guides)
        results.append(pv)

    # 取 top3
    top3 = sorted(results, key=lambda x: -x.score)[:3]
    # 可视化
    def viz(g):
        cols = g.shape[1]
        header = "   " + " ".join(f"C{c+1:>3}" for c in range(cols)) + "\n"
        sep = "  +" + "----"*cols + "+\n"
        body = ""
        for r in range(g.shape[0]):
            row = "|".join(f"{'' if g[r,c]==-1 else g[r,c]:>3}" for c in range(cols))
            body += f"R{r+1:>2}|{row}|\n" + sep
        return "```lua\n"+header+sep+body+"```"

    return {
        "status": "success",
        "visual_grid": viz(grid),
        "results": {
            int(pv.value): {"pos": pv.pos, "score": round(pv.score,4), "py_guide": pv.py_guide}
            for pv in top3
        }
    }
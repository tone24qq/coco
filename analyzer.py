# analyzer.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
import numpy as np
import concurrent.futures

app = FastAPI(title="ScratchCard Analyzer")

# —— 数据模型 —— #
@dataclass
class EvalContext:
    grid: np.ndarray      # 整张表
    row: int              # 候选行
    col: int              # 候选列
    val: int              # 候选数字
    rows: int             # 行数
    cols: int             # 列数
    max_val: int          # 最大数字（rows*cols）
    used: set             # 已出现数字集合

class ProposedValue(BaseModel):
    pos: Tuple[int,int]               # (row, col)，0-base
    value: int
    score: float
    rationale: str                    # 各模块打分概要
    modules: Dict[str, float]         # 每个模块的加权得分

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]]         # -1 表示空格
    top_k: int = Field(3, ge=1, le=20)  # 返回前K名

class AnalyzeResponse(BaseModel):
    status: str
    visual_grid: str
    results: List[ProposedValue]

# —— 可视化表格 —— #
def visualize_grid(grid: List[List[int]]) -> str:
    R = len(grid); C = len(grid[0]) if R else 0
    col_labels = "   " + "".join(f" C{c+1:<3}" for c in range(C)) + "\n"
    sep = "  +" + "+".join("----" for _ in range(C)) + "+\n"
    body = ""
    for r,row in enumerate(grid):
        cells = "|".join(f"{(str(v) if v>=0 else ''):>3} " for v in row)
        body += f"R{r+1:<2}|{cells}|\n" + sep
    return "```lua\n" + col_labels + sep + body + "```"

# —— 规则字典：name → (fn(ctx)->float, weight) —— #
# 返回值是 0.0–1.0 分数，最后乘以 weight
RULES: Dict[str, Tuple[Callable[[EvalContext], float], float]] = {
    "A6_fixed_empty":    (lambda ctx: 1.0 if ctx.grid[ctx.row,ctx.col] == -1 else 0.0, 1.0),
    "M3_interval":       (lambda ctx: min([abs(ctx.col-i) for i,v in enumerate(ctx.grid[ctx.row]) if v==ctx.val], default=0)/max(ctx.cols-1,1), 1.2),
    "A9_diagonal":       (lambda ctx: 1.0 if ctx.row==ctx.col and ctx.grid[ctx.row,ctx.col]==ctx.val else 0.0, 0.8),
    "M5_row_incr":       (lambda ctx: 1.0 if all(x<y for x,y in zip([v for v in ctx.grid[ctx.row] if v>=0],[w for w in ctx.grid[ctx.row] if w>=0][1:])) else 0.0, 1.1),
    "M10_edge":          (lambda ctx: 1.0 if ctx.row in (0,ctx.rows-1) or ctx.col in (0,ctx.cols-1) else 0.0, 0.7),
    "M13_center_diag":   (lambda ctx: 1.0 if abs(ctx.row-ctx.rows//2)==abs(ctx.col-ctx.cols//2) else 0.0, 0.9),
    "M14_mirror_gap":    (lambda ctx: 1.0 if abs(ctx.val - int(ctx.grid[ctx.row,ctx.cols-1-ctx.col] or -999))<=2 else 0.0, 0.6),
    "M15_parity_block":  (lambda ctx: 1.0 if ((ctx.val%2==0) == ((ctx.row+ctx.col)%2==0)) else 0.0, 0.5),
    # …… 更多自定义规则放这里
}

def _fallback_ilp(ctx: EvalContext) -> float:
    """
    ILP / OR-Tools 精确求解占位：
    将全局问题建模后返回一个综合得分，暂返回0。
    """
    return 0.0

# —— 分析主函数 —— #
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # 转 numpy
    try:
        grid = np.array(req.new_card, dtype=int)
    except:
        raise HTTPException(400, "new_card 无法转为整数字阵")

    if grid.ndim!=2:
        raise HTTPException(400, "new_card 需为二维列表")
    rows, cols = grid.shape
    max_val = rows * cols

    # 边界检查
    if any(len(row)!=cols for row in req.new_card):
        raise HTTPException(400, "表格列长不一致")
    used = set(grid.flatten()) - {-1}

    # 生成所有候选 (r,c,val)
    tasks: List[EvalContext] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r,c] != -1: continue
            for v in range(1, max_val+1):
                if v in used: continue
                tasks.append(EvalContext(grid, r, c, v, rows, cols, max_val, used))

    # 并发评估
    results: List[ProposedValue] = []
    def eval_one(ctx: EvalContext) -> ProposedValue:
        total = 0.0; details = {}
        for name,(fn,wt) in RULES.items():
            sc = fn(ctx) * wt
            total += sc
            details[name] = sc
        # ILP 占位
        ilp_sc = _fallback_ilp(ctx) * 1.5
        total += ilp_sc
        details["ILP"] = ilp_sc

        # 归一化
        score = total / (len(RULES) + 1)
        # 理由串
        rationale = ",".join(f"{n}={details[n]:.2f}" for n in details if details[n]>0)
        return ProposedValue(pos=(ctx.row,ctx.col), value=ctx.val, score=score,
                             modules=details, rationale=rationale)

    with concurrent.futures.ThreadPoolExecutor() as exe:
        for pv in exe.map(eval_one, tasks):
            results.append(pv)

    # 选 TopK
    results.sort(key=lambda x: -x.score)
    topk = results[:req.top_k]

    return AnalyzeResponse(
        status="success",
        visual_grid=visualize_grid(req.new_card),
        results=topk
    )
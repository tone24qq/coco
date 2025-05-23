# analyzer.py

import os, json
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

# —— 1. 载入样本 & 预计算 memory_frequency —— #
BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(BASE_DIR, "memory_cards.json"), "r", encoding="utf-8") as f:
    memory_samples = json.load(f)["memory_cards"]

# memory_freq[(r,c)][val] = 出现次数
memory_freq: Dict[Tuple[int,int], Counter] = defaultdict(Counter)
for grid in memory_samples:
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v != -1:
                memory_freq[(r, c)][v] += 1

# memory_total[(r,c)] = 有效样本总数
memory_total: Dict[Tuple[int,int], int] = {
    pos: sum(cnt.values()) for pos, cnt in memory_freq.items()
}

def m0_memory_frequency(grid, pos: List[int], value: int) -> float:
    """M0: 基于样本频率的打分， freq/total"""
    pos = tuple(pos)
    total = memory_total.get(pos, 0)
    if total == 0:
        return 0.0
    freq = memory_freq[pos].get(value, 0)
    return freq / total

# —— 2. 你的原有死逻辑 & 其它规则 —— #
def a6_fixed_position(grid, pos, value): return grid[pos[0]][pos[1]] == -1
def m3_interval_consistency(grid, pos, value):
    r,c=pos; intervals=[abs(c-i) for i,v in enumerate(grid[r]) if v==value]
    return intervals[0] if intervals else 0
def a9_diagonal_symmetry(grid,pos,value): r,c=pos; return r==c and grid[r][c]==value
def m5_sequence_direction(grid,pos,value):
    row=[v for v in grid[pos[0]] if v!=-1]; return all(x<y for x,y in zip(row,row[1:]))
def m10_edge_prediction(grid,pos,value):
    rows,cols=len(grid),len(grid[0]); r,c=pos
    return int(r in (0,rows-1) or c in (0,cols-1))
def m11_jump_zone(grid,pos,value): return int((pos[0]+pos[1])%2==0)
def m12_diagonal_repeat(grid,pos,value): r,c=pos; return int(r==c and grid[r][c]==value)
def m13_center_rotation(grid,pos,value):
    rows,cols=len(grid),len(grid[0]); cr,cc=rows//2,cols//2
    return int(abs(pos[0]-cr)==abs(pos[1]-cc))
def m14_mirror_diff(grid,pos,value):
    r,c=pos; cols=len(grid[0]); m=cols-1-c
    if 0<=m<cols and grid[r][m]!=-1:
        return int(abs(grid[r][c]-grid[r][m])<=2)
    return 0
def m15_parity_block(grid,pos,value):
    return int((value%2==0)==((pos[0]+pos[1])%2==0))
def m16_upper_lower_ratio(grid,pos,value):
    return int(pos[0] < len(grid)//2)
def m17_column_mirror_reverse(grid,pos,value):
    r,c=pos; cols=len(grid[0]); return int(grid[r][c]==grid[r][cols-1-c])
def m18_horizontal_delta(grid,pos,value):
    r,c=pos; cols=len(grid[0])
    if 0<c<cols-1:
        L,R=grid[r][c-1],grid[r][c+1]
        if L!=-1 and R!=-1:
            return int(abs(L-R)<=2)
    return 0
def m19_arc_shape(grid,pos,value):
    return int(abs(pos[0]-pos[1])<=1)

RULES = [
    a6_fixed_position, m3_interval_consistency, a9_diagonal_symmetry,
    m5_sequence_direction, m10_edge_prediction, m11_jump_zone,
    m12_diagonal_repeat, m13_center_rotation, m14_mirror_diff,
    m15_parity_block, m16_upper_lower_ratio, m17_column_mirror_reverse,
    m18_horizontal_delta, m19_arc_shape
]

# —— 3. Models —— #
class ProposedValue(BaseModel):
    pos: List[int]
    value: int
    score: float = 0.0
    py_guide: str

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

# —— 4. 格子可视化 —— #
def visualize_grid(grid):
    rows,cols=len(grid),len(grid[0])
    header = "   " + "".join(f"C{c+1:>4}" for c in range(cols)) + "\n"
    sep = "  +" + "+".join("----" for _ in range(cols)) + "+\n"
    body = ""
    for r,row in enumerate(grid):
        cells = "|".join(f"{v if v!=-1 else '':>4}" for v in row)
        body += f"R{r+1:<2}|{cells}|\n" + sep
    return "```lua\n" + header + sep + body + "```"

# —— 5. 分析 Endpoint —— #
app = FastAPI()

@app.post(
    "/analyze",
    operation_id="analyze",
    summary="刮刮卡候选补格",
    tags=["analysis"]
)
def analyze(req: AnalyzeRequest):
    grid = req.new_card
    # 1) 验证
    if not grid or not req.proposed_values:
        return {"status": "error", "message": "new_card 或 proposed_values 缺失"}
    # 2) 对每个候选，重新打分：逻辑+记忆
    α, β = 0.6, 0.4  # 逻辑 vs 样本权重
    for pv in req.proposed_values:
        # 规则打分：所有 RULES 求平均
        rule_score = sum(fn(grid, pv.pos, pv.value) for fn in RULES) / len(RULES)
        mem_score  = m0_memory_frequency(grid, pv.pos, pv.value)
        pv.score   = α * rule_score + β * mem_score

    # 3) 排序取 top3
    top3 = sorted(req.proposed_values, key=lambda x: -x.score)[:3]
    return {
        "status": "success",
        "visual_grid": visualize_grid(grid),
        "results": [
            {"pos": v.pos, "value": v.value, "score": v.score, "py_guide": v.py_guide}
            for v in top3
        ]
    }

# —— 6. 修正 OpenAPI Schema —— #
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="Scratch Card Analyzer",
        version="1.0.0",
        description="结合逻辑 + 样本频率的刮刮卡补格分析",
        routes=app.routes,
    )
    schema["openapi"] = "3.1.0"
    schema["servers"] = [
        {"url": "https://<你的域名或 Render URL>", "description": "Production"},
        {"url": "http://localhost:8000",             "description": "Local"},
    ]
    schema["paths"]["/analyze"]["post"]["operationId"] = "analyze"
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi
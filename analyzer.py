from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()

class ProposedValue(BaseModel):
    pos: List[int]
    value: int
    score: float
    py_guide: str

class SimplifiedAnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

def visualize_grid(grid: List[List[int]]) -> str:
    rows, cols = len(grid), len(grid[0]) if grid else 0
    header = "   " + "".join(f" C{c+1:<3}" for c in range(cols)) + "\n"
    sep = "  +" + "+".join("----" for _ in range(cols)) + "+\n"
    body = ""
    for r, row in enumerate(grid):
        vals = "|".join(f"{v if v!=-1 else '':>3} " for v in row)
        body += f"R{r+1:<2}|{vals}|\n" + sep
    return "```lua\n" + header + sep + body + "```"

# —— ① 原有死逻辑模块 —— #
def a6_fixed_position(grid, pos, value):          return grid[pos[0]][pos[1]] == -1
def m3_interval_consistency(grid, pos, value):
    r,c=pos; ints=[abs(c-i) for i in range(len(grid[r])) if grid[r][i]==value]
    return ints[0] if ints else -1
def a9_diagonal_symmetry(grid,pos,value):         r,c=pos; return r==c and grid[r][c]==value
def m5_sequence_direction(grid,pos,value):
    row=[v for v in grid[pos[0]] if v!=-1]
    return all(x<y for x,y in zip(row,row[1:]))
def m10_edge_prediction(grid,pos,value):
    r,c=pos; R,C=len(grid),len(grid[0])
    return r in (0,R-1) or c in (0,C-1)
def m11_jump_zone(grid,pos,value):                r,c=pos; return (r+c)%2==0
def m12_diagonal_repeat(grid,pos,value):          r,c=pos; return r==c and grid[r][c]==value
def m13_center_rotation(grid,pos,value):
    R,C=len(grid),len(grid[0]); cr,cc=R//2,C//2; r,c=pos
    return abs(r-cr)==abs(c-cc)
def m14_mirror_diff(grid,pos,value):
    R,C=len(grid),len(grid[0]); r,c=pos; mc=C-1-c
    return (mc>=0 and grid[r][mc]!=-1 and abs(grid[r][c]-grid[r][mc])<=2)
def m15_parity_block(grid,pos,value):             r,c=pos; return (value%2==0)==((r+c)%2==0)
def m16_upper_lower_ratio(grid,pos,value):        return pos[0]<len(grid)//2
def m17_column_mirror_reverse(grid,pos,value):
    R,C=len(grid),len(grid[0]); r,c=pos
    return grid[r][c]==grid[r][C-1-c]
def m18_horizontal_delta(grid,pos,value):
    r,c=pos; C=len(grid[0])
    if 0<c<C-1:
        a,b=grid[r][c-1],grid[r][c+1]
        return a!=-1 and b!=-1 and abs(a-b)<=2
    return False
def m19_arc_shape(grid,pos,value):                r,c=pos; return abs(r-c)<=1

ORIG_RULES = [
    ("A6", a6_fixed_position),
    ("M3", m3_interval_consistency),
    ("A9", a9_diagonal_symmetry),
    ("M5_row_inc", m5_sequence_direction),
    ("M10_edge", m10_edge_prediction),
    ("M11_jump", m11_jump_zone),
    ("M12_diag_rep", m12_diagonal_repeat),
    ("M13_ctr_rot", m13_center_rotation),
    ("M14_mirror", m14_mirror_diff),
    ("M15_parity", m15_parity_block),
    ("M16_uratio", m16_upper_lower_ratio),
    ("M17_col_mir", m17_column_mirror_reverse),
    ("M18_hdelta", m18_horizontal_delta),
    ("M19_arc", m19_arc_shape),
]

# —— ② 公式探测模块 —— #
def detect_linear_mod(grid: np.ndarray):
    pts, vals = [], []
    R,C = grid.shape
    for r in range(R):
        for c in range(C):
            v = grid[r,c]
            if v!=-1:
                pts.append([r,c,1]); vals.append(v)
    if len(pts)<3: return None
    A = np.array(pts, float); y = np.array(vals, float)
    coeffs,*_ = np.linalg.lstsq(A,y,rcond=None)
    pred = A.dot(coeffs)
    if np.allclose(pred,y,1e-6):
        return ("linear", coeffs)
    rounded = np.round(pred).astype(int)
    for M in range(2, R*C+1):
        if np.all(rounded % M == y.astype(int)):
            return ("linear_mod", coeffs, M)
    return None

@app.post("/analyze")
def analyze(req: SimplifiedAnalyzeRequest):
    # 校验
    if not req.new_card or not req.proposed_values:
        return {"status":"error","message":"缺少必要欄位"}
    if any(len(r)!=len(req.new_card[0]) for r in req.new_card):
        return {"status":"error","message":"表格列長度不一致"}

    grid = req.new_card
    # 1) 应用原有规则给初始 score 小幅打分
    for name, fn in ORIG_RULES:
        for pv in req.proposed_values:
            try:
                if fn(grid, pv.pos, pv.value):
                    pv.score += 1.0
                    pv.py_guide += f",{name}"
            except:
                pass

    # 2) 公式探测叠加  +5 分大幅提升
    npg = np.array(grid, int)
    fm = detect_linear_mod(npg)
    if fm:
        kind = fm[0]
        if kind=="linear":
            a,b,c0 = fm[1]
            for pv in req.proposed_values:
                r,c = pv.pos
                if abs(a*r + b*c + c0 - pv.value)<1e-6:
                    pv.score += 5.0
                    pv.py_guide += ",FORMULA(linear)"
        else:
            a,b,c0,M = fm[1], fm[2]
            a,b,c0 = a
            for pv in req.proposed_values:
                r,c = pv.pos
                if int(round(a*r + b*c + c0))%M == pv.value:
                    pv.score += 5.0
                    pv.py_guide += f",FORMULA(mod{M})"

    # 3) 最终 Top3
    top3 = sorted(req.proposed_values, key=lambda x:-x.score)[:3]
    return {
        "status":"success",
        "visual_grid": visualize_grid(grid),
        "results": {
            top3[0].value: [
                {"pos":pv.pos, "score":round(pv.score,4), "py_guide":pv.py_guide}
                for pv in top3
            ]
        }
    }
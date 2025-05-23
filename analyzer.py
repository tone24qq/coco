from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()

class ProposedValue(BaseModel):
    pos: List[int]       # [row, col]
    value: int
    score: float = 0.0   # 會動態計算
    py_guide: str = ""   # 模組命中記錄

class SimplifiedAnalyzeRequest(BaseModel):
    new_card: List[List[int]]        # -1 表示空格
    proposed_values: List[ProposedValue]

def visualize_grid(grid: List[List[int]]) -> str:
    # 保留原始視覺化函式，不動
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    col_labels = "   " + "".join(f" C{c+1:<3}" for c in range(cols)) + "\n"
    sep = "  +" + "+".join("----" for _ in range(cols)) + "+\n"
    body = ""
    for r, row in enumerate(grid):
        row_vals = "|".join(f"{str(v) if v!=-1 else '':>3} " for v in row)
        body += f"R{r+1:<2}|{row_vals}|\n" + sep
    return "```lua\n" + col_labels + sep + body + "```"

@app.post("/analyze")
def analyze(req: SimplifiedAnalyzeRequest):
    # 1) 基本檢查
    if not req.new_card or not req.proposed_values:
        return {"status":"error","message":"缺少 new_card 或 proposed_values"}
    if any(len(r)!=len(req.new_card[0]) for r in req.new_card):
        return {"status":"error","message":"表格列長度不一致"}

    # 2) 懶載入、一次轉 ndarray 並量化 dtype
    import numpy as np
    grid = np.array(req.new_card, dtype=np.int16)
    rows, cols = grid.shape
    center = (rows//2, cols//2)

    # 3) 定義各模組的向量化/快速檢查
    def eval_pv(pv: ProposedValue):
        r, c = pv.pos
        v = pv.value
        sc = 0.0
        guides = []

        # A6: fixed position 必須原本是空
        if grid[r, c] == -1:
            sc += 1.0; guides.append("A6")

        # M5: row 遞增序列
        row_vals = grid[r, grid[r]!=-1]
        if row_vals.size>1 and np.all(np.diff(row_vals)>0):
            sc += 1.0; guides.append("M5")

        # M3: interval consistency (越靠近越優)
        same_cols = np.where(grid[r]==v)[0]
        if same_cols.size>0:
            dist = abs(c - same_cols[0])
            sc += max(0.0, 1.0 - dist/cols)
            guides.append(f"M3(d={dist})")

        # M10: border
        if r==0 or r==rows-1 or c==0 or c==cols-1:
            sc += 0.5; guides.append("M10")

        # M11: chess-zone
        if (r+c)%2==0:
            sc += 0.2; guides.append("M11")

        # M13: diagonal from center
        if abs(r-center[0])==abs(c-center[1]):
            sc += 0.5; guides.append("M13")

        # M14: mirror diff <=2
        mc = cols-1-c
        if 0<=mc<cols and grid[r,c]!=-1 and grid[r,mc]!=-1 and abs(int(grid[r,c])-int(grid[r,mc]))<=2:
            sc += 0.5; guides.append("M14")

        # M15: parity block
        if (v%2==0)==((r+c)%2==0):
            sc += 0.2; guides.append("M15")

        pv.score = sc
        pv.py_guide = ",".join(guides)
        return pv

    # 4) 使用 ThreadPoolExecutor(2) 並行計算（手機端發熱最小化）
    with ThreadPoolExecutor(max_workers=2) as exe:
        futures = [exe.submit(eval_pv, pv) for pv in req.proposed_values]
        evaluated = [f.result() for f in as_completed(futures)]

    # 5) 取前 3 名
    top3 = sorted(evaluated, key=lambda x: -x.score)[:3]
    visual = visualize_grid(req.new_card)

    # 6) 回傳
    return {
        "status": "success",
        "visual_grid": visual,
        "results": [
            {
                "value": pv.value,
                "pos": pv.pos,
                "score": round(pv.score, 4),
                "py_guide": pv.py_guide
            } for pv in top3
        ]
    }
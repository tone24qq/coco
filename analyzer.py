from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class ProposedValue(BaseModel):
    pos: List[int]  # [row, col]
    value: int
    score: float
    py_guide: str

class SimplifiedAnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

def visualize_grid(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    col_labels = "   " + "".join([f" C{c+1:<3}" for c in range(cols)]) + "\n"
    sep = "  +" + "+".join(["----"] * cols) + "+\n"
    body = ""
    for r, row in enumerate(grid):
        # -1 轉成空白顯示
        row_vals = "|".join(f"{str(val) if val != -1 else '':>3} " for val in row)
        body += f"R{r+1:<2}|{row_vals}|\n" + sep
    return "```lua\n" + col_labels + sep + body + "```"

@app.post("/analyze")
def analyze(req: SimplifiedAnalyzeRequest):
    try:
        # 進階異常檢查
        if not req.new_card or not req.proposed_values:
            return {"status": "error", "message": "缺少必要欄位"}
        # 驗證每列長度一致
        if any(len(row) != len(req.new_card[0]) for row in req.new_card):
            return {"status": "error", "message": "表格列長度不一致"}

        # 取最高三個score排序
        sorted_results = sorted(req.proposed_values, key=lambda x: -x.score)[:3]
        visual_grid = visualize_grid(req.new_card)

        return {
            "status": "success",
            "visual_grid": visual_grid,
            "results": {
                sorted_results[0].value: [
                    {
                        "pos": r.pos,
                        "score": round(r.score, 4),
                        "py_guide": r.py_guide
                    } for r in sorted_results
                ]
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
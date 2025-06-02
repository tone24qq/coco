from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Any
import numpy as np
from analyzer import analyze_full_board

class CombinedInput(BaseModel):
    new_card: dict[str, Any]
    grid: list[list[int]]

    model_config = {
        # 允許欄位值中包含任意型別，以免 Pydantic 無法為 Any 生成 schema
        "arbitrary_types_allowed": True
    }

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Service is running"}

@app.head("/")
async def root_head():
    return {}

@app.post("/analyze")
async def analyze(input: CombinedInput = Body(...)):
    the_new_card = input.new_card
    grid = input.grid

    # 基本驗證：檢查 grid 不為空且為矩形，且大小不超過 100×100
    if grid is None:
        raise HTTPException(status_code=422, detail="No grid data provided.")
    row_lengths = [len(row) for row in grid]
    if len(set(row_lengths)) != 1:
        raise HTTPException(status_code=422, detail="Grid rows are not of equal length.")
    n_rows, n_cols = len(grid), len(grid[0]) if grid else 0
    if n_rows == 0 or n_cols == 0:
        raise HTTPException(status_code=422, detail="Grid must be non-empty 2D array.")
    if n_rows > 100 or n_cols > 100:
        raise HTTPException(status_code=422, detail=f"Grid size {n_rows}x{n_cols} exceeds 100x100 limit.")

    # 轉成 NumPy 陣列並檢查維度
    try:
        arr = np.array(grid, dtype=int)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid grid data type: {e}")
    if arr.ndim != 2:
        raise HTTPException(status_code=422, detail="Grid data is not a 2D matrix.")

    # 執行分析
    try:
        score_matrix = analyze_full_board(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return {
        "new_card_received": the_new_card,
        "score_matrix": score_matrix.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
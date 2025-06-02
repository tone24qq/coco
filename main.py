# main.py

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, root_validator
from typing import Any, List, Union
import os
import numpy as np
from analyzer import analyze_full_board

class NewCardModel(BaseModel):
    """
    這個 Model 可以接受兩種形式：
      1. new_card: List[List[int]]
      2. new_card: {"grid": List[List[int]]}
    最終都會把底下的 .grid 屬性設成正確的二維 int 清單。
    """
    grid: List[List[int]] = []

    @root_validator(pre=True)
    def allow_list_or_dict(cls, values):
        """
        如果前端直接把 new_card 塞成 List[List[int]]，
        該 validator 會把它轉成 {"grid": <that list>}。
        """
        # 如果前端 new_card 本身就是一個 list of lists，就直接把它視為 grid
        if isinstance(values, list):
            return {"grid": values}
        # 如果前端傳 { "grid": […] } 就照原樣
        if "grid" in values:
            return {"grid": values["grid"]}
        # 其餘情況算錯誤
        raise ValueError("new_card 必須是 二維陣列，或包含 grid 欄位的物件")

class CombinedInput(BaseModel):
    new_card: NewCardModel
    proposed_values: List[Any] = []

    model_config = {
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
    # 這裡 input.new_card.grid 一定是一個 List[List[int]]
    grid = input.new_card.grid

    # 一樣做原本的檢查：矩形、不超過 100x100、轉 np.array…… 
    if grid is None:
        raise HTTPException(status_code=422, detail="new_card.grid 不可為空。")
    row_lengths = [len(row) for row in grid]
    if len(set(row_lengths)) != 1:
        raise HTTPException(status_code=422, detail="Grid rows are not of equal length.")
    n_rows, n_cols = len(grid), len(grid[0]) if grid else 0
    if n_rows == 0 or n_cols == 0:
        raise HTTPException(status_code=422, detail="Grid 必須為非空 2D 陣列。")
    if n_rows > 100 or n_cols > 100:
        raise HTTPException(status_code=422, detail=f"Grid 大小 {n_rows}x{n_cols} 超過 100x100 限制。")

    try:
        arr = np.array(grid, dtype=int)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid grid data type: {e}")
    if arr.ndim != 2:
        raise HTTPException(status_code=422, detail="Grid data is not a 2D matrix.")

    try:
        score_matrix = analyze_full_board(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # 結果（示範回 top3、used_modules）
    flat = score_matrix.flatten()
    idx_sorted = np.argsort(-flat)
    top3_idx = idx_sorted[:3].tolist()
    top3_coords = [(int(i // n_cols) + 1, int(i % n_cols) + 1) for i in top3_idx]

    return {
        "new_card_received": {"grid": grid},
        "proposed_values_received": input.proposed_values,
        "top3_positions": top3_coords,
        "used_modules": getattr(analyze_full_board, "USED_MODULES", [])
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
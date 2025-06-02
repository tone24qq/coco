from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Any, List
import os
import numpy as np
from analyzer import analyze_full_board

# 先定義 new_card 裡面真正要的欄位
class NewCardModel(BaseModel):
    grid: List[List[int]]

    # 如果你希望後續能接受任意型別，也可加上 arbitrary_types_allowed
    model_config = {
        "arbitrary_types_allowed": True
    }

# 定義整體請求結構：new_card、以及 proposed_values（可選）
class CombinedInput(BaseModel):
    new_card: NewCardModel
    proposed_values: List[Any] = []  # 如果前端不傳也沒關係，預設為空列表

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
    """
    接收的 JSON 範例必須長這樣：
    {
      "new_card": {
        "grid": [[...], [...], ...]
      },
      "proposed_values": [...]
    }
    """

    # 1. 把 grid 拿出來
    grid = input.new_card.grid

    # 2. 基本驗證：檢查 grid 不為空、為矩形、大小不超 100x100
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

    # 3. 轉成 NumPy 陣列並檢查維度
    try:
        arr = np.array(grid, dtype=int)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid grid data type: {e}")
    if arr.ndim != 2:
        raise HTTPException(status_code=422, detail="Grid data is not a 2D matrix.")

    # 4. 呼叫分析函式
    try:
        score_matrix = analyze_full_board(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # 5. 回傳結果：包含原始 new_card、proposed_values，以及分析出的 top3（由後端自行計算）
    #    這裡假設 analyze_full_board 回的是整張分數矩陣，你可以自行在這裡算 top3
    flat = score_matrix.flatten()
    # 找出三個最大分數的索引（以行主序優先），示範作法如下：
    idx_sorted = np.argsort(-flat)  # 由大到小排序
    top3_idx = idx_sorted[:3].tolist()
    # 轉成 (row, col) 形式
    top3_coords = [(int(i // n_cols) + 1, int(i % n_cols) + 1) for i in top3_idx]

    return {
        # 原樣回傳前端傳過來的 new_card、proposed_values：
        "new_card_received": input.new_card.dict(),
        "proposed_values_received": input.proposed_values,
        # 回傳前 3 名分數最高的位置（1-based row/col）：
        "top3_positions": top3_coords,
        # 回傳這次計算時實際用到的模組列表（示範用，需你在 analyze_full_board 裡回傳或記錄）
        # 假設 analyze_full_board 裡維護了一個 global list: USED_MODULES
        "used_modules": getattr(analyze_full_board, "USED_MODULES", [])
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
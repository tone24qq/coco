from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import numpy as np
from analyzer import analyze_full_board

# 定義輸入資料的模型，用於 FastAPI 請求的自動解析
class GridInput(BaseModel):
    grid: list[list[int]]  # 二維整數列表表示的數字盤面

app = FastAPI()

@app.post("/analyze")
async def analyze(input: GridInput = Body(...)):
    """
    接收包含數字盤面的請求，執行分析並回傳分數矩陣。
    """
    # 提取盤面資料並進行基本驗證
    grid = input.grid
    if grid is None:
        raise HTTPException(status_code=422, detail="No grid data provided.")
    # 確認盤面是矩形（每列長度相等）
    row_lengths = [len(row) for row in grid]
    if len(set(row_lengths)) != 1:
        raise HTTPException(status_code=422, detail="Grid rows are not of equal length.")
    # 確認盤面尺寸在允許範圍內 (最大 100x100)
    n_rows, n_cols = len(grid), len(grid[0]) if grid else 0
    if n_rows == 0 or n_cols == 0:
        raise HTTPException(status_code=422, detail="Grid must be non-empty 2D array.")
    if n_rows > 100 or n_cols > 100:
        raise HTTPException(status_code=422, detail=f"Grid size {n_rows}x{n_cols} exceeds 100x100 limit.")
    # 將輸入轉換為 NumPy 二維整數陣列
    try:
        arr = np.array(grid, dtype=int)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid grid data type: {e}")
    if arr.ndim != 2:
        raise HTTPException(status_code=422, detail="Grid data is not a 2D matrix.")
    # 呼叫分析函式進行盤面分析
    try:
        score_matrix = analyze_full_board(arr)
    except Exception as e:
        # 捕捉分析中未預期的錯誤
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    # 將結果轉為Python列表並回傳
    return {"score_matrix": score_matrix.tolist()}

# 如果直接執行此模組（例如在本地測試環境），啟動 FastAPI 服務
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
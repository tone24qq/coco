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
    new_card: Union[List[List[int]], dict]

    @root_validator(pre=True)
    def normalize_grid(cls, values: dict) -> dict:
        nc = values.get("new_card")
        if isinstance(nc, dict) and "grid" in nc:
            values["new_card"] = nc["grid"]
        if not isinstance(values["new_card"], list):
            raise ValueError("new_card must be a 2D list or a dict containing 'grid'")
        return values

    @property
    def grid(self) -> np.ndarray:
        """
        轉成 numpy 2D array，並檢查格式。
        - 不能有 None、不能有 nan
        - 只能是 int
        - 形狀要矩形（每行長度相同）
        """
        arr = np.array(self.new_card, dtype=int)
        if arr.ndim != 2:
            raise ValueError("grid must be 2-dimensional")
        # 檢查每行長度一致
        rows = [len(row) for row in self.new_card]
        if any(r != rows[0] for r in rows):
            raise ValueError("all rows in grid must have the same length")
        return arr

class AnalyzeRequest(BaseModel):
    """
    這個 Model 只用於 /analyze 路由，接受：
      - grid: List[List[int]]
    """
    grid: List[List[int]]

    @root_validator(pre=True)
    def check_grid(cls, values: dict) -> dict:
        g = values.get("grid")
        if not isinstance(g, list):
            raise ValueError("grid must be a 2D list of integers")
        # 檢查是否是矩形格式
        if any(not isinstance(row, list) for row in g):
            raise ValueError("grid must be a 2D list")
        row_lengths = [len(row) for row in g]
        if any(l != row_lengths[0] for l in row_lengths):
            raise ValueError("all rows in grid must have the same length")
        return values

app = FastAPI()

@app.post("/new_card")
def create_card(model: NewCardModel):
    try:
        grid = model.grid
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # 在此可以做一些「登錄新卡」的動作，比如存檔、存資料庫
    # 目前簡單回傳確認
    return {"status": "received", "shape": grid.shape}

@app.post("/analyze")
def analyze_card(request: AnalyzeRequest):
    try:
        grid = np.array(request.grid, dtype=int)
    except Exception:
        raise HTTPException(status_code=400, detail="grid must be a 2D list of integers")
    # grid 內部：「-1」代表未開
    # 呼叫 analyzer.py 中的 analyze_full_board，回傳每個未開位置的分數
    try:
        scores = analyze_full_board(grid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"scores": scores}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# 若要以指令列方式執行，也可以在此執行 uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
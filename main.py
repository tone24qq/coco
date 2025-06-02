# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from analyzer import Analyzer  # 確保 analyzer.py 和 main.py 在同一資料夾

class AnalyzeRequest(BaseModel):
    """
    分析請求：
      - grid: 二維整數列表（隱藏格以 -1 表示）
      - target: 要查找的數字
    """
    grid: List[List[int]]
    target: int

app = FastAPI()
analyzer = Analyzer()

@app.post("/analyze")
def do_analyze(request: AnalyzeRequest):
    # 驗證並轉成 numpy 陣列
    try:
        grid_arr = np.array(request.grid, dtype=int)
    except Exception:
        raise HTTPException(status_code=400, detail="grid 必須是二維整數列表")

    # 呼叫 Analyzer 進行分析
    try:
        scores = analyzer.analyze(grid_arr, request.target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析過程出錯：{e}")

    return {"scores": scores}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
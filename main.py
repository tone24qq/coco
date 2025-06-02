# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from analyzer import Analyzer  # 确保 analyzer.py 与 main.py 在同一目录

class AnalyzeRequest(BaseModel):
    """
    分析请求模型：
      - grid: 二维整数列表（隐藏格以 -1 表示）
      - target: 要查找的数字
    """
    grid: List[List[int]]
    target: int

app = FastAPI()
analyzer = Analyzer()

@app.post("/analyze")
def do_analyze(request: AnalyzeRequest):
    # 验证并转换为 numpy 数组
    try:
        grid_arr = np.array(request.grid, dtype=int)
    except Exception:
        raise HTTPException(status_code=400, detail="grid 必须是二维整数列表")

    # 调用 Analyzer 进行分析
    try:
        scores = analyzer.analyze(grid_arr, request.target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析过程出错：{e}")

    # 返回 {位置ID: 分数} 的字典
    return {"scores": scores}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
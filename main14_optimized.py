# main14_optimized.py

import os
import numpy as np
import asyncio
import concurrent.futures
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Tuple, Any

from analyzer11_optimized import analyze_with_prior  # 請確保這個函式已按照先前建議做了輸入檢查

app = FastAPI()

# ---------- CORS 設定（如有需要，可保留或移除） ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 單筆分析 Request/Response Model ----------
class AnalyzeRequest(BaseModel):
    grid: List[List[int]] = Field(
        ...,
        description="二維整數矩陣，空格請用 -1 表示。例如 [[-1, 5], [3, -1]]"
    )
    target: int = Field(..., ge=0, description="欲分析的目標數字（非負整數）")
    request_id: str = Field("single_req", description="可選的請求識別字串")

class AnalyzeResponse(BaseModel):
    positions: List[Tuple[int, int, float]] = Field(
        ...,
        description="前三個最推薦的 (row, col, confidence) 三元組"
    )

# ---------- 多筆 Batch 分析 Request/Response Model ----------
class AnalyzeBatchRequest(BaseModel):
    grids: List[List[List[int]]] = Field(
        ...,
        description="一組二維整數矩陣清單。每個矩陣都必須含有 -1 代表空格。"
    )
    targets: List[int] = Field(
        ...,
        description="與 `grids` 一一對應的目標數字清單（每張卡片的 target）。"
    )
    request_id: str = Field("batch_req", description="可選的 batch 請求識別字串")

# ---------- 共用的 Grid 驗證函式 ----------
def validate_grid(grid_list: Any) -> np.ndarray:
    """
    將傳入的 List[List[int]] 轉成 np.ndarray，並做格式檢查：
      1. 必須是 2D 列表
      2. 陣列元素必須都是整數
      3. 必須至少含有一個 -1（代表空格）
      4. 不能出現 0 或空值來代表空格
    驗證通過後回傳 np.ndarray，若不符合則拋 HTTPException。
    """
    try:
        arr = np.array(grid_list)
    except Exception:
        raise HTTPException(status_code=422, detail="格式錯誤：傳入值無法轉成 numpy 陣列。")

    # 1. 檢查是否 2D
    if arr.ndim != 2:
        raise HTTPException(status_code=422, detail="格式錯誤：grid 必須是二維陣列 (2D)。")

    # 2. 檢查是否整數型態
    if not np.issubdtype(arr.dtype, np.integer):
        raise HTTPException(status_code=422, detail="格式錯誤：grid 中的所有元素必須是整數。")

    # 3. 檢查是否至少含有一個 -1
    if not np.any(arr == -1):
        raise HTTPException(
            status_code=422,
            detail="格式錯誤：grid 中必須至少含有一個 -1 來表示空格。如果你把空格設成 0，請改用 -1。"
        )

    # 4. 檢查不能用 0 代表空格（若出現 0，但 0 也可能是有效數字，僅做警示）
    #    依使用場景而定，可改成嚴格禁止 0 或僅當 0 不在候選值範圍內才警告。
    #    這裡示範如果完全不允許 0 出現：
    if np.any(arr == 0):
        raise HTTPException(
            status_code=422,
            detail="格式錯誤：請勿用 0 代表空格，空格請統一用 -1。"
        )

    return arr

# ---------- 單筆分析 Endpoint ----------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    # Step 1: 先做基本檢查
    grid_arr = validate_grid(request.grid)

    # Step 2: 檢查 target
    if request.target < 0:
        raise HTTPException(status_code=422, detail="格式錯誤：target 必須是非負整數。")

    # Step 3: 呼叫 CPU 密集型的分析函式
    try:
        # 如果想非同步，也可改成：
        # result = await asyncio.to_thread(analyze_with_prior, grid_arr, request.target, request.request_id)
        result = analyze_with_prior(grid_arr, request.target, request.request_id)
    except ValueError as ve:
        # 若 analyze_with_prior 本身也做了輸入檢查並拋 ValueError，可一併捕捉
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # 其他意外錯誤
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤：{e}")

    return AnalyzeResponse(positions=result)

# ---------- Batch 分析 Endpoint（同時並行處理多張卡） ----------
@app.post("/analyze/batch", response_model=List[AnalyzeResponse])
async def analyze_batch(request: AnalyzeBatchRequest) -> List[AnalyzeResponse]:
    # Step 1: 長度檢查
    if len(request.grids) != len(request.targets):
        raise HTTPException(status_code=422, detail="grids 與 targets 長度必須一致。")

    # Step 2: 逐一驗證並轉成 numpy.ndarray
    num_cards = len(request.grids)
    np_grids: List[np.ndarray] = []
    for idx, grid_list in enumerate(request.grids):
        try:
            arr = validate_grid(grid_list)
        except HTTPException as he:
            # 如果第 idx 張卡就有問題，回傳明確錯誤
            raise HTTPException(
                status_code=422,
                detail=f"第 {idx+1} 張卡的格式有誤：{he.detail}"
            )
        np_grids.append(arr)

    # Step 3: 驗證 targets
    for idx, tgt in enumerate(request.targets):
        if not isinstance(tgt, int) or tgt < 0:
            raise HTTPException(
                status_code=422,
                detail=f"第 {idx+1} 個 target 必須是非負整數，目前是：{tgt}"
            )

    # Step 4: 並行呼叫 analyze_with_prior
    results: List[Any] = [None] * num_cards
    loop = asyncio.get_running_loop()

    # 可自行調整 max_workers 至較適合的值 (例如 CPU 核心數)
    max_workers = min(4, num_cards)  # 最多開 4 線程，或卡片數量少就用對應數量

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for i in range(num_cards):
            # 每張卡用不同 request_id（方便 log 區分）
            rid = f"{request.request_id}_{i+1}"
            tasks.append(
                loop.run_in_executor(
                    executor,
                    analyze_with_prior,
                    np_grids[i],
                    request.targets[i],
                    rid
                )
            )
        # 等待所有並行任務完成
        completed = await asyncio.gather(*tasks, return_exceptions=True)

    # Step 5: 檢查是否有任何異常
    responses: List[AnalyzeResponse] = []
    for idx, res in enumerate(completed):
        if isinstance(res, Exception):
            # 如果其中某張卡分析過程出錯，就回傳對應錯誤
            # 但完整設計可改成「部分成功，部分失敗」的回應結構
            raise HTTPException(
                status_code=500,
                detail=f"第 {idx+1} 張卡分析失敗：{res}"
            )
        # 若成功，就包成 AnalyzeResponse
        responses.append(AnalyzeResponse(positions=res))

    return responses

# ---------- 簡單的 Health Check Endpoint ----------
@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": asyncio.get_event_loop().time()}


# --------------- 若有需要，可直接啟動此檔案 ---------------
# uvicorn main14_optimized:app --host 0.0.0.0 --port 10000
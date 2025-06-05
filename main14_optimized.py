# main14_optimized.py

import os
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
import numpy as np
import asyncio
import concurrent.futures
import time
import psutil
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Any

from analyzer11_optimized import analyze_with_prior

# Environment constants
# TODO: 若要修改值，請透過環境變數設定
MAX_WORKERS = int(os.getenv("MAX_WORKERS", os.cpu_count()))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 10))

app = FastAPI()

# ---------- CORS 設定（如有需要，可保留或移除） ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 根路由：同時支援 GET 和 HEAD ----------
@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    print("✅ / route is working")
    # FastAPI/Starlette 會自動在 HEAD 請求時剃掉 body
    return {"status": "ok", "message": "Service is alive."}


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

    # 4. 檢查不能用 0 代表空格
    if np.any(arr == 0):
        raise HTTPException(
            status_code=422,
            detail="格式錯誤：請勿用 0 代表空格，空格請統一用 -1。"
        )

    return arr


def _validate_and_convert_batch(request: AnalyzeBatchRequest) -> Tuple[List[np.ndarray], List[int]]:
    """Validate and convert batch request grids and targets.
    
    Args:
        request (AnalyzeBatchRequest): Batch request object.
        
    Returns:
        Tuple[List[np.ndarray], List[int]]: Validated grids and targets.
        
    Raises:
        HTTPException: If validation fails.
    """
    if len(request.grids) != len(request.targets):
        raise HTTPException(status_code=422, detail="grids 與 targets 長度必須一致。")

    np_grids = []
    for idx, grid_list in enumerate(request.grids):
        try:
            arr = validate_grid(grid_list)
        except HTTPException as he:
            raise HTTPException(status_code=422, detail=f"第 {idx+1} 張卡的格式有誤：{he.detail}")
        np_grids.append(arr)

    for idx, tgt in enumerate(request.targets):
        if not isinstance(tgt, int) or tgt < 0:
            raise HTTPException(status_code=422, detail=f"第 {idx+1} 個 target 必須是非負整數，目前是：{tgt}")

    return np_grids, request.targets


async def _run_batch_tasks(grids: List[np.ndarray], targets: List[int], request_id: str) -> List[AnalyzeResponse]:
    """Run batch analysis tasks with timeout and parallelism.
    
    Args:
        grids (List[np.ndarray]): List of validated grid arrays.
        targets (List[int]): List of target numbers.
        request_id (str): Base request ID for logging.
        
    Returns:
        List[AnalyzeResponse]: List of analysis responses.
        
    Raises:
        HTTPException: If any task fails or times out.
    """
    num_cards = len(grids)
    results = [None] * num_cards
    loop = asyncio.get_running_loop()

    max_workers = min(MAX_WORKERS, num_cards)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for i in range(num_cards):
            rid = f"{request_id}_{i+1}"
            tasks.append(
                asyncio.wait_for(
                    loop.run_in_executor(executor, analyze_with_prior, grids[i], targets[i], rid),
                    timeout=30.0
                )
            )
        completed = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    for idx, res in enumerate(completed):
        if isinstance(res, asyncio.TimeoutError):
            raise HTTPException(
                status_code=500,
                detail=f"第 {idx+1} 張卡分析超時 (30秒)"
            )
        elif isinstance(res, Exception):
            raise HTTPException(
                status_code=500,
                detail=f"第 {idx+1} 張卡分析失敗：{res}"
            )
        responses.append(AnalyzeResponse(positions=res))

    return responses


# ---------- 單筆分析 Endpoint ----------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    start_time = time.time()
    logger.info(f"[{request.request_id}] Request received: grid={len(request.grid)}x{len(request.grid[0])}, target={request.target}, CPU={psutil.cpu_percent():.1f}%")
    
    try:
        grid_arr = validate_grid(request.grid)
        if request.target < 0:
            raise HTTPException(status_code=422, detail="格式錯誤：target 必須是非負整數。")

        result = await asyncio.to_thread(analyze_with_prior, grid_arr, request.target, request.request_id)
        end_time = time.time()
        logger.info(f"[{request.request_id}] Analysis completed in {end_time - start_time:.4f} seconds, CPU={psutil.cpu_percent():.1f}%")
        return AnalyzeResponse(positions=result)
    except Exception as e:
        end_time = time.time()
        logger.error(f"[{request.request_id}] Analysis failed in {end_time - start_time:.4f} seconds: {e}")
        return JSONResponse(
            status_code=500 if not isinstance(e, HTTPException) else e.status_code,
            content={
                "code": str(e.status_code) if isinstance(e, HTTPException) else "500",
                "message": str(e),
                "detail": str(e.__cause__) if hasattr(e, "__cause__") else None
            }
        )


# ---------- Batch 分析 Endpoint ----------
@app.post("/analyze/batch", response_model=List[AnalyzeResponse])
async def analyze_batch(request: AnalyzeBatchRequest) -> List[AnalyzeResponse]:
    start_time = time.time()
    logger.info(f"[{request.request_id}] Batch request received: {len(request.grids)} cards, CPU={psutil.cpu_percent():.1f}%")
    
    try:
        grids, targets = _validate_and_convert_batch(request)
        responses = await _run_batch_tasks(grids, targets, request.request_id)
        end_time = time.time()
        logger.info(f"[{request.request_id}] Batch analysis completed in {end_time - start_time:.4f} seconds, CPU={psutil.cpu_percent():.1f}%")
        return responses
    except Exception as e:
        end_time = time.time()
        logger.error(f"[{request.request_id}] Batch analysis failed in {end_time - start_time:.4f} seconds: {e}")
        return JSONResponse(
            status_code=500 if not isinstance(e, HTTPException) else e.status_code,
            content={
                "code": str(e.status_code) if isinstance(e, HTTPException) else "500",
                "message": str(e),
                "detail": str(e.__cause__) if hasattr(e, "__cause__") else None
            }
        )


# ---------- 簡單的 Health Check Endpoint ----------
@app.get("/health")
def health_check():
    start_time = time.time()
    try:
        return {"status": "ok", "timestamp": asyncio.get_event_loop().time()}
    except Exception as e:
        end_time = time.time()
        logger.error(f"Health check failed in {end_time - start_time:.4f} seconds: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "code": "500",
                "message": str(e),
                "detail": str(e.__cause__) if hasattr(e, "__cause__") else None
            }
        )

@app.on_event("startup")
def on_startup():
    logging.getLogger(__name__).info("[Startup] Instantiating VectorizedBrainModules to load heatmap…")
    VectorizedBrainModules()
    logging.getLogger(__name__).info("[Startup] VectorizedBrainModules instantiation complete.")

# Middleware for concurrency limit
@app.middleware("http")
async def limit_concurrency(request, call_next):
    if psutil.cpu_percent() > 90 or len(asyncio.all_tasks()) > MAX_CONCURRENT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"code": "429", "message": "Too Many Requests", "detail": "Server is overloaded"}
        )
    response = await call_next(request)
    return response


# --------------- 若有需要，可直接啟動此檔案 ---------------
# uvicorn main14_optimized:app --host 0.0.0.0 --port 10000
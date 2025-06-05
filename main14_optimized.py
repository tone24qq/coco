"""
main14_optimized.py - Fully fixed FastAPI with keep-alive, /, HEAD support
"""
import time
import logging
import asyncio
import os
from typing import List
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, validator

from analyzer11_optimized import analyze_with_prior

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main14_optimized")

# === Metrics ===
startup_time = datetime.now()
request_count = 0
total_process_time = 0.0
keep_alive_task_ref = None

# === Background keep-alive ===
async def keep_alive_task():
    port = os.getenv("PORT", "10000")
    client = httpx.AsyncClient(timeout=5.0)
    while True:
        try:
            url = f"http://127.0.0.1:{port}/health"
            response = await client.get(url)
            logger.info(f"Keep-alive: status={response.status_code}")
        except Exception as e:
            logger.warning(f"Keep-alive failed (non-fatal): {e}")
        await asyncio.sleep(200)

# === Models ===
class AnalyzeRequest(BaseModel):
    grid: List[List[int]]
    target: int
    top_k: int = Field(3, ge=1, le=10)

    @validator('grid')
    def validate_grid(cls, v):
        if not v or not v[0]:
            raise ValueError("Grid cannot be empty")
        if not all(len(row) == len(v[0]) for row in v):
            raise ValueError("Grid must be rectangular")
        return v

class Position(BaseModel):
    row: int
    col: int
    confidence: float

class AnalyzeResponse(BaseModel):
    positions: List[Position]
    grid_shape: tuple[int, int]
    process_time: float

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_requests: int
    average_process_time: float
    modules_count: int

# === Lifecycle hook ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    global keep_alive_task_ref
    logger.info("Starting FastAPI application")
    keep_alive_task_ref = asyncio.create_task(keep_alive_task())
    yield
    keep_alive_task_ref.cancel()
    logger.info("Shutting down FastAPI application")

# === App init ===
app = FastAPI(
    title="Number Card AI Analysis Service",
    version="2.0.0",
    description="Stable endpoint with internal metrics and auto-recovery",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# === Root / Endpoint ===
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root():
    return PlainTextResponse("橘子 AI 分析服務已啟動")

# === Request counter ===
@app.middleware("http")
async def count_requests(request: Request, call_next):
    global request_count
    request_count += 1
    return await call_next(request)

# === Health check ===
@app.get("/health", response_model=HealthResponse)
async def health_check():
    from vectorized_modules import SCORING_MODULES
    uptime = (datetime.now() - startup_time).total_seconds()
    avg = total_process_time / request_count if request_count else 0
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        total_requests=request_count,
        average_process_time=avg,
        modules_count=len(SCORING_MODULES)
    )

# === Analyze ===
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    global total_process_time

    try:
        grid = np.array(request.grid, dtype=np.int32)
        if not np.any(grid == -1):
            raise HTTPException(status_code=400, detail="No blank cells")

        start = time.time()
        results = analyze_with_prior(grid, request.target, request_id=str(request_count))
        duration = time.time() - start
        total_process_time += duration

        positions = [
            Position(row=r, col=c, confidence=conf)
            for r, c, conf in results[:request.top_k]
        ]
        return AnalyzeResponse(positions=positions, grid_shape=grid.shape, process_time=duration)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")

# === Batch analyze ===
@app.post("/analyze/batch")
async def analyze_batch(requests: List[AnalyzeRequest]):
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Batch limit exceeded")

    results = []
    for req in requests:
        try:
            result = await analyze(req)
            results.append({"success": True, "data": result.dict()})
        except HTTPException as e:
            results.append({"success": False, "error": e.detail})
    return {"results": results}

# === Global error handler ===
@app.exception_handler(Exception)
async def all_errors(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Unhandled exception"})

# === Local run ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main14_optimized:app", host="0.0.0.0", port=8014, reload=True)
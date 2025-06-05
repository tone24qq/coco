"""
main14_optimized.py - Optimized FastAPI with robust keep-alive, health, and analysis modules
"""

import os
import time
import logging
import asyncio
from typing import List
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, validator

from analyzer11_optimized import analyze_with_prior

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

startup_time = datetime.now()
request_count = 0
total_process_time = 0.0

async def keep_alive_task(port: str):
    """Ping /health every 200s to prevent idle shutdown"""
    url = f"http://127.0.0.1:{port}/health"
    client = httpx.AsyncClient(timeout=5.0)

    # Wait until service is fully ready
    for i in range(5):
        try:
            response = await client.get(url)
            if response.status_code == 200:
                logger.info(f"Keep-alive initial OK at attempt {i+1}")
                break
        except Exception:
            pass
        await asyncio.sleep(1)

    # Periodic ping
    while True:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                logger.info("Keep-alive: status=200 OK")
            else:
                logger.warning(f"Keep-alive non-200: {response.status_code}")
        except Exception as e:
            logger.warning(f"Keep-alive failed (non-fatal): {e}")
        await asyncio.sleep(200)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle, starts keep-alive task"""
    logger.info("Starting FastAPI application")
    port = os.getenv("PORT", "10000")
    task = asyncio.create_task(keep_alive_task(port))
    yield
    task.cancel()
    logger.info("Shutting down FastAPI application")

app = FastAPI(
    title="Number Card AI Analysis Service",
    description="Efficient position recommendation with vectorized 4-module analysis",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root():
    return PlainTextResponse("橘子 AI 分析服務正常運行中 🍊", status_code=200)

class AnalyzeRequest(BaseModel):
    grid: List[List[int]] = Field(..., description="2D grid, -1 for blanks")
    target: int = Field(..., description="Target number")
    top_k: int = Field(3, ge=1, le=10, description="Return top K results")

    @validator('grid')
    def validate_grid(cls, v):
        if not v or not v[0]:
            raise ValueError("Grid cannot be empty")
        rows = len(v)
        cols = len(v[0])
        if not all(len(row) == cols for row in v):
            raise ValueError("Grid must be rectangular")
        return v

class Position(BaseModel):
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)

class AnalyzeResponse(BaseModel):
    positions: List[Position]
    grid_shape: tuple[int, int]
    process_time: float

class HealthResponse(BaseModel):
    status: str = Field("healthy")
    uptime_seconds: float
    total_requests: int
    average_process_time: float
    modules_count: int

@app.middleware("http")
async def count_requests(request, call_next):
    global request_count
    request_count += 1
    response = await call_next(request)
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = (datetime.now() - startup_time).total_seconds()
    avg_time = total_process_time / request_count if request_count else 0
    from vectorized_modules import SCORING_MODULES
    return HealthResponse(
        uptime_seconds=uptime,
        total_requests=request_count,
        average_process_time=avg_time,
        modules_count=len(SCORING_MODULES)
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    global total_process_time
    try:
        grid = np.array(request.grid, dtype=np.int32)
        if not np.any(grid == -1):
            raise HTTPException(status_code=400, detail="No blank cells in grid")

        start_time = time.time()
        results = analyze_with_prior(grid, request.target, request_id=str(request_count))
        process_time = time.time() - start_time
        total_process_time += process_time

        positions = [
            Position(row=r, col=c, confidence=conf)
            for r, c, conf in results[:request.top_k]
        ]
        return AnalyzeResponse(
            positions=positions,
            grid_shape=grid.shape,
            process_time=process_time
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze/batch")
async def analyze_batch(requests: List[AnalyzeRequest]):
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Batch requests limited to 10")

    results = []
    for req in requests:
        try:
            result = await analyze(req)
            results.append({"success": True, "data": result.dict()})
        except HTTPException as e:
            results.append({"success": False, "error": e.detail})
    return {"results": results}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.on_event("startup")
async def startup_info():
    from vectorized_modules import SCORING_MODULES
    logger.info("Service started")
    logger.info(f"Loaded {len(SCORING_MODULES)} scoring modules")
    logger.info("API docs: http://localhost:10000/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main14_optimized:app",
        host="0.0.0.0",
        port=10000,
        reload=True,
        log_level="info"
    )
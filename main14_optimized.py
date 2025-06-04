"""
main14_optimized.py - 優化版主程式，更簡潔的 API 設計
"""

import os
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# 導入優化版分析器

from analyzer11_optimized import (
analyze_with_prior,
_load_memory_folder,
compute_weights_from_memory,
GLOBAL_WEIGHTS,
SHAPE_WEIGHTS,
_score_cache
)

# 設定日誌

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(**name**)

# 全域變數

startup_time = datetime.now()
request_count = 0
total_process_time = 0.0

# 背景任務管理

async def periodic_memory_update():
“”“定期更新記憶體資料”””
while True:
await asyncio.sleep(300)  # 每5分鐘
try:
_load_memory_folder()
global GLOBAL_WEIGHTS, SHAPE_WEIGHTS
GLOBAL_WEIGHTS, SHAPE_WEIGHTS = compute_weights_from_memory()
logger.info(“已更新記憶體資料和權重”)
except Exception as e:
logger.error(f”更新記憶體資料失敗: {e}”)

# Pydantic 模型

class AnalyzeRequest(BaseModel):
“”“簡化的分析請求模型”””
grid: List[List[int]] = Field(…, description=“2D網格，-1表示空格”)
target: Optional[int] = Field(None, description=“目標數字”)
top_k: int = Field(3, ge=1, le=10, description=“返回前K個結果”)

```
@validator('grid')
def validate_grid(cls, v):
    if not v or not v[0]:
        raise ValueError("網格不能為空")
    rows = len(v)
    cols = len(v[0])
    if not all(len(row) == cols for row in v):
        raise ValueError("網格必須是矩形")
    return v

class Config:
    schema_extra = {
        "example": {
            "grid": [
                [1, 2, -1],
                [4, -1, 6],
                [-1, 8, 9]
            ],
            "target": 3,
            "top_k": 3
        }
    }
```

class Position(BaseModel):
“”“位置結果模型”””
row: int = Field(…, ge=0, description=“行索引（0-based）”)
col: int = Field(…, ge=0, description=“列索引（0-based）”)
confidence: float = Field(…, ge=0, le=1, description=“信心分數”)

class AnalyzeResponse(BaseModel):
“”“分析響應模型”””
positions: List[Position] = Field(…, description=“推薦位置列表”)
grid_shape: tuple[int, int] = Field(…, description=“網格形狀”)
process_time: float = Field(…, description=“處理時間（秒）”)
cache_hit: bool = Field(False, description=“是否命中快取”)

class HealthResponse(BaseModel):
“”“健康檢查響應”””
status: str = “healthy”
uptime_seconds: float
total_requests: int
average_process_time: float
memory_samples: int
cache_size: int
modules_count: int

# 生命週期管理

@asynccontextmanager
async def lifespan(app: FastAPI):
“”“應用生命週期管理”””
# 啟動時
logger.info(“正在啟動 FastAPI 應用…”)

```
# 創建背景任務
task = asyncio.create_task(periodic_memory_update())

yield

# 關閉時
logger.info("正在關閉 FastAPI 應用...")
task.cancel()
try:
    await task
except asyncio.CancelledError:
    pass
```

# 創建 FastAPI 應用

app = FastAPI(
title=“數獨/數字卡片 AI 分析服務”,
description=“使用向量化計算和歷史學習的高效位置推薦服務”,
version=“2.0.0”,
lifespan=lifespan
)

# CORS 中間件

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

# 請求計數中間件

@app.middleware(“http”)
async def count_requests(request, call_next):
global request_count
request_count += 1
response = await call_next(request)
return response

# API 路由

@app.get(”/health”, response_model=HealthResponse)
async def health_check():
“”“健康檢查端點”””
uptime = (datetime.now() - startup_time).total_seconds()
avg_time = total_process_time / request_count if request_count > 0 else 0

```
from analyzer11_optimized import MEMORY_SAMPLES, SCORING_MODULES

return HealthResponse(
    uptime_seconds=uptime,
    total_requests=request_count,
    average_process_time=avg_time,
    memory_samples=len(MEMORY_SAMPLES),
    cache_size=len(_score_cache),
    modules_count=len(SCORING_MODULES)
)
```

@app.post(”/analyze”, response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
“”“主要分析端點”””
global total_process_time

```
try:
    # 轉換為 numpy array
    grid = np.array(request.grid, dtype=np.int32)
    
    # 檢查是否有空格
    if not np.any(grid == -1):
        raise HTTPException(status_code=400, detail="網格中沒有空格可分析")
    
    # 記錄開始時間
    start_time = time.time()
    
    # 檢查快取
    grid_hash = hash(grid.tobytes())
    cache_hit = grid_hash in _score_cache
    
    # 執行分析
    target = request.target if request.target else -1
    results = analyze_with_prior(grid, target, request_id=str(request_count))
    
    # 計算處理時間
    process_time = time.time() - start_time
    total_process_time += process_time
    
    # 轉換結果
    positions = [
        Position(row=r, col=c, confidence=conf)
        for r, c, conf in results[:request.top_k]
    ]
    
    return AnalyzeResponse(
        positions=positions,
        grid_shape=grid.shape,
        process_time=process_time,
        cache_hit=cache_hit
    )
    
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"分析失敗: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="內部伺服器錯誤")
```

@app.post(”/analyze/batch”)
async def analyze_batch(requests: List[AnalyzeRequest]):
“”“批量分析端點”””
if len(requests) > 10:
raise HTTPException(status_code=400, detail=“批量請求最多支援10個”)

```
results = []
for req in requests:
    try:
        result = await analyze(req)
        results.append({"success": True, "data": result.dict()})
    except HTTPException as e:
        results.append({"success": False, "error": e.detail})

return {"results": results}
```

@app.get(”/stats”)
async def get_stats():
“”“獲取服務統計資訊”””
from analyzer11_optimized import _cache_hits, _cache_misses, MEMORY_SAMPLES

```
cache_total = _cache_hits + _cache_misses
cache_hit_rate = _cache_hits / cache_total if cache_total > 0 else 0

# 統計歷史樣本分佈
shape_distribution = {}
for sample in MEMORY_SAMPLES:
    shape = sample['card_shape']
    shape_key = f"{shape[0]}x{shape[1]}"
    shape_distribution[shape_key] = shape_distribution.get(shape_key, 0) + 1

return {
    "service": {
        "uptime_seconds": (datetime.now() - startup_time).total_seconds(),
        "total_requests": request_count,
        "average_process_time": total_process_time / request_count if request_count > 0 else 0
    },
    "cache": {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": cache_hit_rate,
        "size": len(_score_cache)
    },
    "memory": {
        "total_samples": len(MEMORY_SAMPLES),
        "shape_distribution": shape_distribution,
        "global_weights": GLOBAL_WEIGHTS,
        "shape_specific_weights": {str(k): v for k, v in SHAPE_WEIGHTS.items()}
    }
}
```

@app.post(”/cache/clear”)
async def clear_cache():
“”“清除快取”””
_score_cache.clear()
return {“message”: “快取已清除”}

@app.post(”/memory/reload”)
async def reload_memory(background_tasks: BackgroundTasks):
“”“重新載入記憶體資料”””
def reload_task():
_load_memory_folder()
global GLOBAL_WEIGHTS, SHAPE_WEIGHTS
GLOBAL_WEIGHTS, SHAPE_WEIGHTS = compute_weights_from_memory()

```
background_tasks.add_task(reload_task)
return {"message": "正在背景重新載入記憶體資料"}
```

# 錯誤處理

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
logger.error(f”未處理的異常: {exc}”, exc_info=True)
return JSONResponse(
status_code=500,
content={“detail”: “內部伺服器錯誤”}
)

# 啟動資訊

@app.on_event(“startup”)
async def startup_info():
from analyzer11_optimized import SCORING_MODULES
logger.info(f”服務啟動完成”)
logger.info(f”已載入 {len(SCORING_MODULES)} 個評分模組”)
logger.info(f”API 文檔: http://localhost:8014/docs”)

if **name** == “**main**”:
import uvicorn
uvicorn.run(
“main14_optimized:app”,
host=“0.0.0.0”,
port=8014,
reload=True,
log_level=“info”
)
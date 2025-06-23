import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from logging.handlers import RotatingFileHandler
import asyncio

from analyzer import predict_scratch_card  # 你的核心邏輯

# ───────────────────────────────
# 1. Logging
# ───────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
log_handlers = [
    logging.StreamHandler(sys.stdout),
    RotatingFileHandler("app.log", mode="a", encoding="utf-8",
                        maxBytes=5_000_000, backupCount=3),
]
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger(__name__)

# ───────────────────────────────
# 2. FastAPI 基礎
# ───────────────────────────────
app = FastAPI(
    title="Scratch Card Prediction API",
    version="1.0.0",
    description="Predict hidden numbers in scratch-card grids "
                "with Monte-Carlo + heuristic modules.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ───────────────────────────────
# 3. Schemas
# ───────────────────────────────
class GridRequest(BaseModel):
    grid: List[List[int]]
    target_num: Optional[int] = None
    iterations: Optional[int] = 5000

class Prediction(BaseModel):
    row: int
    col: int
    candidates: List[int]
    probability: float
    reasons: List[str]
    module_scores: Dict[str, float]

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    full_probabilities: Dict[str, Dict[str, float]]

# ───────────────────────────────
# 4. 基本路由
# ───────────────────────────────
startup_time = datetime.utcnow().isoformat() + "Z"

@app.get("/", response_class=JSONResponse, status_code=status.HTTP_200_OK)
async def root() -> Dict[str, Any]:
    return {"status": "OK", "startup": startup_time}

@app.head("/", response_class=PlainTextResponse,
          status_code=status.HTTP_200_OK)
async def root_head() -> str:
    return ""

# ───────────────────────────────
# 5. /predict
# ───────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(req: GridRequest):
    try:
        rows, cols = len(req.grid), len(req.grid[0])
        if rows < 2 or cols < 2:
            raise ValueError("Grid must be at least 2x2.")
        max_val = rows * cols
        known_vals = [v for row in req.grid for v in row if v != -1]
        if len(known_vals) != len(set(known_vals)):
            raise ValueError("Grid contains duplicate numbers.")
        if any(v < 1 or v > max_val for v in known_vals):
            raise ValueError(f"Numbers must be between 1 and {max_val}.")

        logger.info("Predict | %dx%d | target=%s | iter=%d",
                    rows, cols, req.target_num, req.iterations)

        result = predict_scratch_card(
            grid=req.grid,
            target_num=req.target_num,
            iterations=req.iterations,
        )

        # 序列化 full_probabilities
        if isinstance(fp := result.get("full_probabilities"), dict):
            clean_fp: Dict[str, Dict[str, float]] = {}
            for (r, c), num_map in fp.items():
                key = f"{r},{c}"
                clean_fp[key] = {str(n): p*100 for n, p in num_map.items()}
            result["full_probabilities"] = clean_fp

        return result

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# ───────────────────────────────
# 6. 非阻塞暖機 (背景執行)
# ───────────────────────────────
async def _background_warm_up():
    dummy_grid = [
        [1, 2, -1, 4, 5],
        [-1, 7, 8, -1, 10],
        [11, -1, 13, 14, -1],
        [-1, 17, 18, -1, 20],
    ]
    try:
        base_iter = int(os.getenv("BASE_ITER", "800")) // 25
        predict_scratch_card(grid=dummy_grid, iterations=base_iter)
        logger.info("Background warm-up done.")
    except Exception as exc:
        logger.error("Warm-up failed: %s", exc, exc_info=True)

@app.on_event("startup")
async def startup_event():
    # 如果不想暖機，把 ENABLE_WARMUP=0
    if os.getenv("ENABLE_WARMUP", "1") == "1":
        asyncio.create_task(_background_warm_up())
    logger.info("Startup handler finished (port now open).")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API shutting down.")

# ───────────────────────────────
# 7. CLI 執行（本地測試用）
# ───────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=LOG_LEVEL.lower(),
    )
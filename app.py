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

from analyzer import predict_scratch_card  # 你自己的核心函式

# ──────────────────────────────────────────────
# 1️⃣ Logging 設定 ── 可用環境變數 LOG_LEVEL 控制
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

log_handlers = [
    logging.StreamHandler(sys.stdout),
    RotatingFileHandler(
        "app.log", mode="a", encoding="utf-8",
        maxBytes=5_000_000, backupCount=3
    ),
]
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 2️⃣ FastAPI 初始化 + CORS
# ──────────────────────────────────────────────
app = FastAPI(
    title="Scratch Card Prediction API",
    version="1.0.0",
    description="Predict hidden numbers in scratch-card grids with Monte-Carlo + heuristic modules.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# 3️⃣ Pydantic Schemas
# ──────────────────────────────────────────────
class GridRequest(BaseModel):
    grid: List[List[int]]
    target_num: Optional[int] = None
    iterations: Optional[int] = 5000

class Prediction(BaseModel):
    row: int
    col: int
    candidates: List[int]
    probability: float           # 0-100 %
    reasons: List[str]
    module_scores: Dict[str, float]

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    full_probabilities: Dict[str, Dict[str, float]]  # 位置字串 -> 數字 -> 機率 %

# ──────────────────────────────────────────────
# 4️⃣ 健康檢查 & 基本路由
# ──────────────────────────────────────────────
startup_time = datetime.utcnow().isoformat() + "Z"

@app.get("/", response_class=JSONResponse, status_code=status.HTTP_200_OK)
async def root() -> Dict[str, Any]:
    return {"status": "OK", "startup": startup_time}

@app.head("/", response_class=PlainTextResponse, status_code=status.HTTP_200_OK)
async def root_head() -> str:
    return ""

# ──────────────────────────────────────────────
# 5️⃣ /predict 端點
# ──────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(req: GridRequest):
    try:
        # --- 基本驗證 ---
        if not req.grid or not all(isinstance(r, list) for r in req.grid):
            raise ValueError("Invalid grid format: expected List[List[int]].")
        rows, cols = len(req.grid), len(req.grid[0])
        if rows < 2 or cols < 2:
            raise ValueError("Grid must be at least 2x2.")
        max_val = rows * cols
        known_vals = [v for row in req.grid for v in row if v != -1]
        if len(known_vals) != len(set(known_vals)):
            raise ValueError("Grid contains duplicate numbers.")
        if any(v < 1 or v > max_val for v in known_vals):
            raise ValueError(f"Numbers must be between 1 and {max_val}.")

        logger.info(
            "Predict API called | size=%dx%d | target=%s | iterations=%d",
            rows, cols, str(req.target_num), req.iterations
        )

        # --- 呼叫核心函式 ---
        result = predict_scratch_card(
            grid=req.grid,
            target_num=req.target_num,
            iterations=req.iterations
        )

        # --- 序列化 full_probabilities ---
        if isinstance(result.get("full_probabilities"), dict):
            clean_fp: Dict[str, Dict[str, float]] = {}
            for loc_key, prob_map in result["full_probabilities"].items():
                # 位置轉字串
                if isinstance(loc_key, (tuple, list)) and len(loc_key) == 2:
                    key_str = f"{loc_key[0]},{loc_key[1]}"
                else:
                    key_str = str(loc_key)

                # 內層數字轉字串、機率轉 %
                inner = {str(num): float(p) * 100 for num, p in prob_map.items()}
                clean_fp[key_str] = inner
            result["full_probabilities"] = clean_fp

        return result

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# ──────────────────────────────────────────────
# 6️⃣ 啟動 & 關閉事件（warm-up）
# ──────────────────────────────────────────────
@app.on_event("startup")
async def warm_up():
    dummy_grid = [
        [1,  2, -1,  4,  5],
        [-1, 7,  8, -1, 10],
        [11, -1, 13, 14, -1],
        [-1, 17, 18, -1, 20],
    ]
    try:
        base_iter = int(os.getenv("BASE_ITER", "800")) // 25
        predict_scratch_card(grid=dummy_grid, iterations=base_iter)
        logger.info("Warm-up completed successfully.")
    except Exception as exc:
        logger.error("Warm-up failed: %s", exc, exc_info=True)
        logger.warning("Continuing startup despite warm-up failure.")

@app.on_event("shutdown")
async def shutdown():
    logger.info("API shutting down to save resources.")

# ──────────────────────────────────────────────
# 7️⃣ 主程式入口 ── 用 uvicorn.run，吃 $PORT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))          # Render 會自動塞 PORT
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level=LOG_LEVEL.lower(),
    )
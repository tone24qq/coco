# app.py  ✅ 可直接覆蓋
import os
port = int(os.getenv("PORT", "10000"))
uvicorn.run(app, host="0.0.0.0", port=port)
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analyzer import predict_scratch_card

# ---------------------------------------------------------------------
# Logging：終端 + 旋轉檔案，防止雲端磁碟被撐爆
# ---------------------------------------------------------------------
from logging.handlers import RotatingFileHandler

log_handlers = [
    logging.StreamHandler(sys.stdout),
    RotatingFileHandler(
        "app.log", mode="a", encoding="utf-8", maxBytes=2_000_000, backupCount=3
    ),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# FastAPI 初始化 + CORS (必要時可關)
# ---------------------------------------------------------------------
app = FastAPI(
    title="Scratch Card Prediction API",
    version="1.0.0",
    description="Predict hidden numbers in scratch-card grids with Monte-Carlo + heuristic modules.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 若有安全顧慮請換成白名單
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------
class GridRequest(BaseModel):
    grid: List[List[int]]
    iterations: int | None = None


class Prediction(BaseModel):
    row: int
    col: int
    candidates: List[int]
    confidences: List[float]


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    full_probabilities: Dict[str, Dict[int, float]]


# ---------------------------------------------------------------------
# 健康檢查 / 根路由 —— Render 監測會對 / 做 HEAD
# ---------------------------------------------------------------------
startup_time = datetime.utcnow().isoformat() + "Z"


@app.get("/", response_class=JSONResponse, status_code=status.HTTP_200_OK)
async def root() -> Dict[str, Any]:
    """
    Lightweight health check.
    Always returns 200 even when background tasks are忙碌.
    """
    return {"status": "OK", "startup": startup_time}


@app.head("/", response_class=PlainTextResponse, status_code=status.HTTP_200_OK)
async def root_head() -> str:  # HEAD 無 body
    return ""


# ---------------------------------------------------------------------
# /predict 端點
# ---------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(req: GridRequest):
    try:
        if not req.grid or not all(isinstance(row, list) for row in req.grid):
            raise ValueError("Invalid grid format: expected List[List[int]].")

        iterations = (
            req.iterations
            or (int(os.getenv("ITER", 5_000_000)) if os.getenv("USE_FORMULA_ONLY") != "1" else 500_000)
        )

        logger.info(
            "Predict API called | size=%dx%d | iter=%d",
            len(req.grid),
            len(req.grid[0]),
            iterations,
        )

        result = predict_scratch_card(req.grid, iterations)
        return result

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------
# Lifespan 事件：啟動暖機
# ---------------------------------------------------------------------
@app.on_event("startup")
async def warm_up():
    dummy_grid = [[-1 for _ in range(5)] for _ in range(4)]
    iterations = int(os.getenv("ITER", 5_000_000)) // 25
    try:
        predict_scratch_card(dummy_grid)   # 不帶 n_iter
        logger.info("Warm-up completed successfully.")
    except Exception as exc:
        logger.error("Warm-up failed: %s", exc, exc_info=True)

# ---------------------------------------------------------------------
# 全局例外處理 —— 避免 FastAPI 預設回 500 時 body 不一致
# ---------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    收攏未捕捉的例外，保證 JSON 格式回傳。
    """
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


# ---------------------------------------------------------------------
# 自檢報告
# ---------------------------------------------------------------------
# - 語法檢查：通過
# - HEAD / GET "/"：永遠 200 OK
# - 日誌：旋轉檔防爆
# - CORS：預設全開，可按需鎖定
# - 任何未捕捉例外：統一回 JSON 500
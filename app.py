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

from analyzer import predict_scratch_card

# Logging
from logging.handlers import RotatingFileHandler

log_handlers = [
    logging.StreamHandler(sys.stdout),
    RotatingFileHandler("app.log", mode="a", encoding="utf-8", maxBytes=5_000_000, backupCount=3),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger(__name__)

# App
app = FastAPI(
    title="Scratch Card Prediction API",
    version="1.0.0",
    description="Predict hidden numbers in scratch-card grids with Monte-Carlo + heuristic modules.",
)

# CORS for frontend/API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Schemas ====

class GridRequest(BaseModel):
    grid: List[List[int]]
    target_num: Optional[int] = None
    iterations: Optional[int] = 5000

class Prediction(BaseModel):
    row: int
    col: int
    candidates: List[int]
    probability: float  # percentage
    reasons: List[str]
    module_scores: Dict[str, float]

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    full_probabilities: Dict[str, Dict[str, float]]

# ==== Routes ====

startup_time = datetime.utcnow().isoformat() + "Z"
os.environ.setdefault("ITER", "5000")
os.environ.setdefault("TOPK_RERANK", "120")
os.environ.setdefault("LOG_LEVEL", "INFO")

@app.get("/", response_class=JSONResponse, status_code=status.HTTP_200_OK)
async def root() -> Dict[str, Any]:
    return {"status": "OK", "startup": startup_time}

@app.head("/", response_class=PlainTextResponse, status_code=status.HTTP_200_OK)
async def root_head() -> str:
    return ""

@app.get("/debug/ping", response_class=JSONResponse, status_code=200)
async def ping() -> Dict[str, str]:
    return {"ping": "pong"}

@app.post("/predict", response_model=PredictResponse, response_class=JSONResponse, status_code=200)
async def predict(req: GridRequest):
    try:
        # 格式檢查
        if not req.grid or not all(isinstance(row, list) for row in req.grid):
            raise ValueError("Invalid grid format: expected List[List[int]].")

        rows, cols = len(req.grid), len(req.grid[0])
        if rows < 2 or cols < 2:
            raise ValueError("Grid must be at least 2x2")

        max_val = rows * cols
        known_vals = [v for row in req.grid for v in row if v != -1]
        if len(known_vals) != len(set(known_vals)):
            raise ValueError("Grid contains duplicate numbers")
        if any(v < 1 or v > max_val for v in known_vals):
            raise ValueError(f"Numbers must be between 1 and {max_val}")

        logger.info(
            "Predict API called | size=%dx%d | target=%s | iterations=%d",
            rows, cols, str(req.target_num), req.iterations
        )

        # 推理邏輯
        result = predict_scratch_card(
            grid=req.grid,
            target_num=req.target_num,
            iterations=req.iterations
        )

        predictions = result.get("predictions", [])
        full_probs = result.get("full_probabilities", {})

        # 整理 full_probabilities: 序列化 key、百分比
        clean_probs = {}
        for loc_key, prob_map in full_probs.items():
            try:
                r, c = loc_key
                key_str = f"{int(r)},{int(c)}"
            except Exception:
                key_str = str(loc_key)
            inner = {}
            for num, prob in prob_map.items():
                try:
                    num_key = str(int(float(num)))
                except Exception:
                    num_key = str(num)
                inner[num_key] = float(prob) * 100
            clean_probs[key_str] = inner

        response_payload = {
            "predictions": predictions,
            "full_probabilities": clean_probs
        }

        return JSONResponse(content=response_payload, status_code=200)

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.on_event("startup")
async def warm_up():
    logger.info("Warm-up disabled to speed up startup.")

@app.on_event("shutdown")
async def shutdown():
    logger.info("API shutting down to save resources.")

# === Launch ===
def run_api() -> None:
    port = int(os.getenv("PORT", "10000"))
    logger.info("Starting API server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    run_api()

app = app
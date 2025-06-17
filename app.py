import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from analyzer import predict_scratch_card

# Logging configuration
from logging.handlers import RotatingFileHandler

log_handlers = [
    logging.StreamHandler(sys.stdout),
    RotatingFileHandler(
        "app.log", mode="a", encoding="utf-8", maxBytes=5_000_000, backupCount=3
    ),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger(__name__)

# FastAPI initialization + CORS
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

# Pydantic schema
class GridRequest(BaseModel):
    grid: List[List[int]]

class Prediction(BaseModel):
    row: int
    col: int
    candidates: List[int]
    confidences: List[float]

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    full_probabilities: Dict[str, Dict[str, float]]

# Health check / root route
startup_time = datetime.utcnow().isoformat() + "Z"

@app.get("/", response_class=JSONResponse, status_code=status.HTTP_200_OK)
async def root() -> Dict[str, Any]:
    return {"status": "OK", "startup": startup_time}

@app.head("/", response_class=PlainTextResponse, status_code=status.HTTP_200_OK)
async def root_head() -> str:
    return ""

@app.post("/predict", response_model=PredictResponse)
async def predict(req: GridRequest):
    try:
        if not req.grid or not all(isinstance(row, list) for row in req.grid):
            raise ValueError("Invalid grid format: expected List[List[int]].")
        logger.info(
            "Predict API called | size=%dx%d",
            len(req.grid),
            len(req.grid[0]),
        )
        result = predict_scratch_card(req.grid, iterations=100000)
        
        if "full_probabilities" in result and isinstance(result["full_probabilities"], dict):
            raw_fp = result["full_probabilities"]
            clean_fp = {}
            for loc_key, prob_map in raw_fp.items():
                try:
                    r, c = loc_key
                    key_str = f"{int(r)},{int(c)}"
                except (TypeError, ValueError):
                    key_str = str(loc_key)
                inner_clean = {}
                for num, p in prob_map.items():
                    try:
                        num_key = str(int(float(num)))
                    except (ValueError, TypeError):
                        num_key = str(num)
                    inner_clean[num_key] = float(p)
                clean_fp[key_str] = inner_clean
            result["full_probabilities"] = clean_fp

        return result
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.on_event("startup")
async def warm_up():
    dummy_grid = [
        [1, 2, -1, 4, 5],
        [-1, 7, 8, -1, 10],
        [11, -1, 13, 14, -1],
        [-1, 17, 18, -1, 20]
    ]
    base_iter = int(os.getenv("BASE_ITER", 100000)) // 25
    try:
        predict_scratch_card(dummy_grid, iterations=base_iter)
        logger.info("Warm-up completed successfully.")
    except Exception as exc:
        logger.error("Warm-up failed: %s", exc, exc_info=True)
        logger.warning("Continuing startup despite warm-up failure.")

@app.on_event("shutdown")
async def shutdown():
    logger.info("API shutting down to save resources.")

if __name__ == "__main__":
    # 移除 run_api 無限循環，交由 Render 觸發
    logger.info("API ready, waiting for Render to handle requests...")
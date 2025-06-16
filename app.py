import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analyzer import predict_scratch_card

# Logging configuration
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
    full_probabilities: Dict[str, Dict[str, float]]  # Updated to match string keys

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
        result = predict_scratch_card(req.grid)
        
        # Serializable Fix: Convert full_probabilities keys to strings
        if "full_probabilities" in result and isinstance(result["full_probabilities"], dict):
            raw_fp = result["full_probabilities"]
            clean_fp = {}
            for loc_key, prob_map in raw_fp.items():
                # Handle outer key (e.g., (np.int64, np.int64) or tuple)
                try:
                    r, c = loc_key
                    key_str = f"{int(r)},{int(c)}"  # Format as "r,c"
                except (TypeError, ValueError):
                    key_str = str(loc_key)  # Fallback for unexpected types

                # Handle inner map (convert numeric keys to strings)
                inner_clean = {}
                for num, p in prob_map.items():
                    try:
                        num_key = str(int(float(num)))  # Convert to int string, handle float
                    except (ValueError, TypeError):
                        num_key = str(num)  # Fallback for non-numeric keys
                    inner_clean[num_key] = float(p)  # Ensure probability is float
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
    base_iter = int(os.getenv("BASE_ITER", 5000)) // 25
    try:
        predict_scratch_card(dummy_grid)
        logger.info("Warm-up completed successfully.")
    except Exception as exc:
        logger.error("Warm-up failed: %s", exc, exc_info=True)
        logger.warning("Continuing startup despite warm-up failure.")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )
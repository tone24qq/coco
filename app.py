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
    iterations: int | None = None

class Prediction(BaseModel):
    row: int
    col: int
    candidates: List[int]
    confidences: List[float]

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    full_probabilities: Dict[str, Dict[int, float]]

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
        iterations = req.iterations or int(os.getenv("ITER", 10_000))
        logger.info(
            "Predict API called | size=%dx%d | iter=%d",
            len(req.grid),
            len(req.grid[0]),
            iterations,
        )
        result = predict_scratch_card(
            req.grid,
            quick_iter=iterations // 2,
            refine_iter=iterations // 2,
            min_total_iter=iterations
        )
        return result
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.on_event("startup")
async def warm_up():
    dummy_grid = [[-1 for _ in range(5)] for _ in range(4)]
    iterations = int(os.getenv("ITER", 5_000)) // 25
    try:
        predict_scratch_card(
            dummy_grid,
            quick_iter=iterations // 2,
            refine_iter=iterations // 2,
            min_total_iter=iterations
        )
        logger.info("Warm-up completed successfully.")
    except Exception as exc:
        logger.error("Warm-up failed: %s", exc, exc_info=True)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

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
    target_num: Optional[int] = None
    iterations: Optional[int] = 5000

class Prediction(BaseModel):
    row: int
    col: int
    candidates: List[int]
    probability: float  # Changed to percentage
    reasons: List[str]  # Added for module contribution reasons
    module_scores: Dict[str, float]

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    full_probabilities: Dict[str, Dict[str, float]]  # String keys for serialization

# Health check / root route
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

@app.post("/predict", response_model=PredictResponse)
async def predict(req: GridRequest):
    try:
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
            len(req.grid), len(req.grid[0]), str(req.target_num), req.iterations
        )
        result = predict_scratch_card(
            grid=req.grid,
            target_num=req.target_num,
            iterations=req.iterations
        )
        
        # Serializable Fix: Convert full_probabilities keys to strings
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
                    inner_clean[num_key] = float(p) * 100  # Convert to percentage
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
    base_iter = int(os.getenv("BASE_ITER", "800")) // 25
    try:
        predict_scratch_card(
            grid=dummy_grid,
            iterations=base_iter
        )
        logger.info("Warm-up completed successfully.")
    except Exception as exc:
        logger.error("Warm-up failed: %s", exc, exc_info=True)
        logger.warning("Continuing startup despite warm-up failure.")

@app.on_event("shutdown")
async def shutdown():
    logger.info("API shutting down to save resources.")

def run_api():
    """Run API with on-demand activation."""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    logger.info("API in sleep mode, will wake on request...")
    while True:
        try:
            server.run()  # 啟動時休眠，呼叫時醒來
            logger.info("API woken up and working...")
        except KeyboardInterrupt:
            server.should_exit = True
            break

if __name__ == "__main__":
    run_api()
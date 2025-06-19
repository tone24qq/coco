import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import psutil
import ray
from fastapi import FastAPI, HTTPException, status
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
    iterations: Optional[int] = 1000

class Prediction(BaseModel):
    row: int
    col: int
    candidates: List[int]
    probability: float
    reasons: List[str]

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

@ray.remote
def predict_task(req: GridRequest):
    """Run prediction task with Ray for parallel processing."""
    try:
        if not req.grid or not all(isinstance(row, list) for row in req.grid):
            raise ValueError("Invalid grid format: expected List[List[int]].")
        rows, cols = len(req.grid), len(req.grid[0])
        if rows < 4 or rows > 20 or cols < 4 or cols > 20:
            raise ValueError("Grid must be 4x4 to 20x20")
        max_val = rows * cols
        known_vals = [v for row in req.grid for v in row if v != -1]
        if len(known_vals) != len(set(known_vals)):
            raise ValueError("Grid contains duplicate numbers")
        if any(v < 1 or v > max_val for v in known_vals):
            raise ValueError(f"Numbers must be between 1 and {max_val}")
        
        logger.info(
            "Predict API called | size=%dx%d | target=%s | iterations=%d | memory=%.1f%% | cpu=%.1f%%",
            rows, cols, str(req.target_num), req.iterations,
            psutil.virtual_memory().percent, psutil.cpu_percent()
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
                    inner_clean[num_key] = float(p) * 100
                clean_fp[key_str] = inner_clean
            result["full_probabilities"] = clean_fp
        
        return result
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/predict", response_model=PredictResponse)
async def predict(req: GridRequest):
    """Handle prediction requests with resource monitoring."""
    safemode_count = 0
    if psutil.virtual_memory().percent > 75 or psutil.cpu_percent() > 90:
        safemode_count += 1
        logger.warning("Entering Safemode: high resource usage (memory=%.1f%%, cpu=%.1f%%, safemode_count=%d)",
                       psutil.virtual_memory().percent, psutil.cpu_percent(), safemode_count)
        req.iterations = max(100, req.iterations // 2)
    
    try:
        result = ray.get(predict_task.remote(req))
        logger.info("Prediction completed | safemode_count=%d", safemode_count)
        return result
    except Exception as exc:
        logger.error("Prediction task failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}") from exc

@app.on_event("startup")
async def warm_up():
    """Warm-up the system with a dummy prediction."""
    try:
        # Ensure Ray initialization with sufficient shared memory
        ray.init(num_cpus=4, object_store_memory=int(8e9 * 0.3))  # 30% of 8GB RAM
        dummy_grid = [
            [1, 2, -1, 4, 5],
            [-1, 7, 8, -1, 10],
            [11, -1, 13, 14, -1],
            [-1, 17, 18, -1, 20]
        ]
        base_iter = int(os.getenv("BASE_ITER", 1000)) // 25
        predict_scratch_card(
            grid=dummy_grid,
            iterations=base_iter
        )
        logger.info("Warm-up completed successfully | memory=%.1f%% | cpu=%.1f%%",
                    psutil.virtual_memory().percent, psutil.cpu_percent())
    except Exception as exc:
        logger.error("Warm-up failed: %s", exc, exc_info=True)
        logger.warning("Continuing startup despite warm-up failure.")

@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on shutdown."""
    ray.shutdown()
    logger.info("API shutting down to save resources.")

def run_api():
    """Run API with on-demand activation."""
    port = int(os.getenv("PORT", 10000))  # Default Render port
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="debug")
    server = uvicorn.Server(config)
    logger.info("API starting on 0.0.0.0:%d | memory=%.1f%% | cpu=%.1f%%",
                port, psutil.virtual_memory().percent, psutil.cpu_percent())
    while True:
        try:
            server.run()
            logger.info("API running successfully")
        except KeyboardInterrupt:
            server.should_exit = True
            break
        except Exception as exc:
            logger.error("API startup failed: %s", exc, exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    run_api()
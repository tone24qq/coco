import base64
import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

# fmt: off
# isort: off
from analyzer import (
    probability_heatmap,
    predict_scratch_card,
    render_heatmap,
)
# isort: on
# fmt: on

# —— Logging setup —————————————————————————————————————————————————————————————
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

# —— FastAPI app & CORS —————————————————————————————————————————————————————————
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

# ==== Schemas ==============================================================


class GridRequest(BaseModel):
    grid: List[List[int]]
    target_num: Optional[int] = None
    iterations: Optional[int] = None
    global_iter: Optional[int] = None
    focus_iter: Optional[int] = None
    top_n: Optional[int] = None
    epsilon: Optional[float] = None


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


class HeatmapRequest(BaseModel):
    grid: List[List[int]]
    k: Optional[int] = None
    iterations: int = 1000
    seed: int = 0
    output_format: str = "base64"


class HeatmapResponse(BaseModel):
    prob_map: Union[List[List[float]], Dict[str, List[List[float]]]]
    heatmap: Optional[str] = None


# ==== Startup & ENV parsing =================================================

startup_time = datetime.utcnow().isoformat() + "Z"

# 默认环境变量（未指定时回落到这些值）
os.environ.setdefault("PHASE1_ITERATIONS", "5000")
os.environ.setdefault("PHASE2_ITERATIONS", "1000")
os.environ.setdefault("PHASE2_TOP_N", "10")
os.environ.setdefault("PHASE2_EPSILON", "0.05")
os.environ.setdefault("LOG_LEVEL", "INFO")

# 解析 ENV 到 Python 常量
PHASE1_ITER  = int(os.getenv("PHASE1_ITERATIONS"))
PHASE2_ITER  = int(os.getenv("PHASE2_ITERATIONS"))
PHASE2_TOP_N = int(os.getenv("PHASE2_TOP_N"))
PHASE2_EPS   = float(os.getenv("PHASE2_EPSILON"))


# ==== Routes ==============================================================


@app.get("/", response_class=JSONResponse, status_code=status.HTTP_200_OK)
async def root() -> Dict[str, Any]:
    return {"status": "OK", "startup": startup_time}


@app.head("/", response_class=PlainTextResponse, status_code=status.HTTP_200_OK)
async def root_head() -> str:
    return ""


@app.get("/debug/ping", response_class=JSONResponse, status_code=200)
async def ping() -> Dict[str, str]:
    return {"ping": "pong"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    response_class=JSONResponse,
    status_code=200,
)
async def predict(req: GridRequest):
    try:
        # — 格式檢查 —
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

        # — 日志记录实际使用的两阶段参数 —
        phase1 = req.iterations or PHASE1_ITER
        phase2 = req.focus_iter or PHASE2_ITER
        top_n  = req.top_n      or PHASE2_TOP_N
        eps    = req.epsilon    or PHASE2_EPS

        logger.info(
            "Predict API called | size=%dx%d | target=%s | phase1=%d | phase2=%d | top_n=%d | eps=%.3f",
            rows,
            cols,
            str(req.target_num),
            phase1,
            phase2,
            top_n,
            eps,
        )

        # — 两阶段推理逻辑 —
        result = predict_scratch_card(
            grid=req.grid,
            target_num=req.target_num,
            iterations=phase1,
            global_iter=req.global_iter,
            focus_iter=phase2,
            top_n=top_n,
            epsilon=eps,
        )

        # — 整理 full_probabilities 为 JSON-safe 格式 —
        predictions = result.get("predictions", [])
        full_probs = result.get("full_probabilities", {})
        clean_probs: Dict[str, Dict[str, float]] = {}
        for loc_key, prob_map in full_probs.items():
            try:
                r, c = loc_key
                key_str = f"{int(r)},{int(c)}"
            except Exception:
                key_str = str(loc_key)
            inner: Dict[str, float] = {}
            for num, prob in prob_map.items():
                num_key = str(int(float(num))) if isinstance(num, (int, float, str)) else str(num)
                inner[num_key] = float(prob) * 100
            clean_probs[key_str] = inner

        response_payload = {
            "predictions": predictions,
            "full_probabilities": clean_probs,
        }
        clean_json = json.loads(json.dumps(response_payload, default=lambda x: float(x)))
        logger.info("✅ Final response payload: %s", clean_json)
        return JSONResponse(content=clean_json, status_code=200)

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail="❌ 回傳 JSON 格式異常：" + str(exc)
        ) from exc


@app.post(
    "/heatmap",
    response_model=HeatmapResponse,
    response_class=JSONResponse,
    status_code=200,
)
async def heatmap(req: HeatmapRequest):
    try:
        prob = probability_heatmap(req.grid, req.k, req.iterations, seed=req.seed)
        if isinstance(prob, dict):
            prob_map = {str(int(k)): v.tolist() for k, v in prob.items()}
            return {"prob_map": prob_map, "heatmap": None}
        if req.output_format.lower() == "raw":
            return {"prob_map": prob.tolist(), "heatmap": None}
        rendered = render_heatmap(prob, req.output_format)
        if isinstance(rendered, bytes):
            b64 = base64.b64encode(rendered).decode("ascii")
        else:
            b64 = rendered
        return {"prob_map": prob.tolist(), "heatmap": b64}
    except Exception as exc:
        logger.error("Heatmap generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.on_event("startup")
async def warm_up():
    logger.info("Warm-up disabled to speed up startup.")


@app.on_event("shutdown")
async def shutdown():
    logger.info("API shutting down to save resources.")


# ==== Launch ===============================================================


def run_api() -> None:
    port = int(os.getenv("PORT", "10000"))
    logger.info("Starting API server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run_api()
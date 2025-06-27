import base64
import logging
import math
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
    result_top_k: Optional[int] = None

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
    target_num: Optional[int] = None
    iterations: int = 1000
    seed: int = 0
    output_format: str = "base64"

class HeatmapResponse(BaseModel):
    prob_map: Union[List[List[float]], Dict[str, List[List[float]]]]
    heatmap: Optional[str] = None

# ==== Startup & ENV parsing =================================================

startup_time = datetime.utcnow().isoformat() + "Z"

os.environ.setdefault("PHASE1_ITERATIONS", "5000")
os.environ.setdefault("PHASE2_ITERATIONS", "1000")
os.environ.setdefault("PHASE2_TOP_N", "10")
os.environ.setdefault("PHASE2_EPSILON", "0.05")
os.environ.setdefault("LOG_LEVEL", "INFO")

PHASE1_ITER = int(os.getenv("PHASE1_ITERATIONS"))
PHASE2_ITER = int(os.getenv("PHASE2_ITERATIONS"))
PHASE2_TOP_N = int(os.getenv("PHASE2_TOP_N"))
PHASE2_EPS = float(os.getenv("PHASE2_EPSILON"))

# ==== Helpers: sanitize floats ===============================================

def safe_float(x: Any) -> float:
    """
    把任何 float / numpy.float / NaN / Inf 變成合法 Python float
    (NaN、+Inf、-Inf 均轉成 0.0)
    """
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v

def sanitize_floats(obj: Any) -> Any:
    """
    遞迴地把 dict/list 裡所有 float 都套 safe_float
    """
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    if isinstance(obj, float):
        return safe_float(obj)
    return obj

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

        # — 日志记录使用参数 —
        phase1 = req.iterations or PHASE1_ITER
        phase2 = req.focus_iter or PHASE2_ITER
        top_n = req.top_n or PHASE2_TOP_N
        eps = req.epsilon or PHASE2_EPS
        top_k = req.result_top_k or int(os.getenv("RESULT_TOP_K", "3"))

        logger.info(
            "Predict API called | size=%dx%d | target=%s | phase1=%d | phase2=%d | top_k=%d | top_n=%d | eps=%.3f",
            rows,
            cols,
            str(req.target_num),
            phase1,
            phase2,
            top_k,
            top_n,
            eps,
        )

        # — 调用核心推理模块 —
        result = predict_scratch_card(
            grid=req.grid,
            target_num=req.target_num,
            iterations=phase1,
            global_iter=req.global_iter,
            focus_iter=phase2,
            top_n=top_n,
            epsilon=eps,
            result_top_k=req.result_top_k,
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
                inner[num_key] = safe_float(prob) * 100
            clean_probs[key_str] = inner

        response_payload = {
            "predictions": predictions,
            "full_probabilities": clean_probs,
        }

        # — 全面 sanitize predictions & full_probabilities —
        safe_payload = sanitize_floats(response_payload)

        logger.info("✅ Final response payload: %s", safe_payload)
        return JSONResponse(content=safe_payload, status_code=200)

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
        # CI 加速：如果 FAST_TEST=1，则迭代次数最多 100
        iters = req.iterations
        if os.getenv("FAST_TEST", "") == "1":
            iters = min(iters, 100)

        # 兼容 k / target_num
        effective_k = req.k if req.k is not None else req.target_num

        # 调用 heatmap 模块
        prob = probability_heatmap(req.grid, effective_k, iters, seed=req.seed)

        # 根据返回类型组织响应
        if isinstance(prob, dict):
            pm = {str(int(k)): v.tolist() for k, v in prob.items()}
            resp = {"prob_map": pm, "heatmap": None}
        elif req.output_format.lower() == "raw":
            resp = {"prob_map": prob.tolist(), "heatmap": None}
        else:
            rendered = render_heatmap(prob, req.output_format)
            b64 = (
                base64.b64encode(rendered).decode("ascii")
                if isinstance(rendered, bytes)
                else rendered
            )
            resp = {"prob_map": prob.tolist(), "heatmap": b64}

        # 过滤掉所有 NaN/Inf
        safe_resp = sanitize_floats(resp)
        return JSONResponse(content=safe_resp, status_code=200)

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
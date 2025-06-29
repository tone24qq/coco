import asyncio
import base64
import json
import logging
import math
import os
import subprocess
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

import brain
# fmt: off
from analyzer import (iter_sample_jsons, predict_scratch_card,
                      probability_heatmap, render_heatmap)

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
HERE = Path(__file__).parent
script = HERE / "excel_cleaner_and_formatter.py"


@app.on_event("startup")
async def pre_startup():  # FIXME rename to avoid F811 duplicate
    logger.info("📂 on_startup: starting Excel cleaning")
    # 用当前 Python 解释器执行清洗脚本
    subprocess.run([sys.executable, str(script)], check=True)
    logger.info("📂 on_startup: cleaning script finished")


@app.get("/debug/priors", response_class=JSONResponse)
async def debug_priors():
    return priors


async def _load_samples_background():
    # 這裡會觸發 analyzer 裡的 logger.info("Total loaded: …")
    for _ in iter_sample_jsons("samples"):
        pass
    logger.info("Sample iteration complete (background)")


@app.on_event("startup")
async def startup_event():
    # 1) 先綁 port（Uvicorn 自行處理）
    # 2) 再排程背景任務
    asyncio.create_task(_load_samples_background())


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
    sample_gamma: Optional[float] = None


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

# 默认环境变量
os.environ.setdefault("PHASE1_ITERATIONS", "5000")
os.environ.setdefault("PHASE2_ITERATIONS", "1000")
os.environ.setdefault("PHASE2_TOP_N", "10")
os.environ.setdefault("PHASE2_EPSILON", "0.05")
os.environ.setdefault("LOG_LEVEL", "INFO")

PHASE1_ITER = int(os.getenv("PHASE1_ITERATIONS"))
PHASE2_ITER = int(os.getenv("PHASE2_ITERATIONS"))
PHASE2_TOP_N = int(os.getenv("PHASE2_TOP_N"))
PHASE2_EPS = float(os.getenv("PHASE2_EPSILON"))

# 全局先验概率（来自 Excel 样本）
priors: Dict[int, float] = {}


# ==== Helpers: sanitize floats ===============================================
def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def sanitize_floats(obj: Any) -> Any:
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
    logger.debug("/predict called with iterations=%s", req.iterations)
    try:
        # 格式校验
        if not req.grid or not all(isinstance(row, list) for row in req.grid):
            raise ValueError("Invalid grid format: expected List[List[int]].")
        rows, cols = len(req.grid), len(req.grid[0])
        if rows < 2 or cols < 2:
            raise ValueError("Grid must be at least 2x2")
        max_val = rows * cols
        known = [v for row in req.grid for v in row if v != -1]
        if len(known) != len(set(known)):
            raise ValueError("Grid contains duplicate numbers")
        if any(v < 1 or v > max_val for v in known):
            raise ValueError(f"Numbers must be between 1 and {max_val}")

        # 参数
        phase1 = req.iterations or PHASE1_ITER
        phase2 = req.focus_iter or PHASE2_ITER
        top_n = req.top_n or PHASE2_TOP_N
        eps = req.epsilon or PHASE2_EPS
        top_k = req.result_top_k or int(os.getenv("RESULT_TOP_K", "3"))

        logger.info(
            "Predict | size=%dx%d | target=%s | ph1=%d | ph2=%d | top_k=%d | top_n=%d | eps=%.3f",
            rows,
            cols,
            str(req.target_num),
            phase1,
            phase2,
            top_k,
            top_n,
            eps,
        )

        # 调用核心推理
        result = predict_scratch_card(
            grid=req.grid,
            target_num=req.target_num,
            iterations=phase1,
            global_iter=req.global_iter,
            focus_iter=phase2,
            top_n=top_n,
            epsilon=eps,
            result_top_k=top_k,
            priors=priors,
            sample_gamma=req.sample_gamma or 0.0,
        )

        # 清洗 full_probabilities
        full_probs = result.get("full_probabilities", {})
        clean_probs: Dict[str, Dict[str, float]] = {}
        for loc, prob_map in full_probs.items():
            key = f"{int(loc[0])},{int(loc[1])}"
            inner: Dict[str, float] = {}
            for num, p in prob_map.items():
                inner[str(int(num))] = safe_float(p) * 100
            clean_probs[key] = inner

        payload = {
            "predictions": result.get("predictions", []),
            "full_probabilities": clean_probs,
            "sample_gamma_used": req.sample_gamma or 0.0,
        }
        safe_payload = sanitize_floats(payload)
        logger.info("✅ Response ready")
        return JSONResponse(content=safe_payload, status_code=200)

    except Exception as exc:
        logger.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/heatmap",
    response_model=HeatmapResponse,
    response_class=JSONResponse,
    status_code=200,
)
async def heatmap(req: HeatmapRequest):
    try:
        iters = (
            min(req.iterations, 100)
            if os.getenv("FAST_TEST") == "1"
            else req.iterations
        )
        k_eff = req.k if req.k is not None else req.target_num
        prob = probability_heatmap(req.grid, k_eff, iters, seed=req.seed)

        if isinstance(prob, dict):
            pm = {str(int(k)): v.tolist() for k, v in prob.items()}
            resp = {"prob_map": pm, "heatmap": None}
        elif req.output_format.lower() == "raw":
            resp = {"prob_map": prob.tolist(), "heatmap": None}
        else:
            img = render_heatmap(prob, req.output_format)
            b64 = base64.b64encode(img).decode() if isinstance(img, bytes) else img
            resp = {"prob_map": prob.tolist(), "heatmap": b64}

        return JSONResponse(content=sanitize_floats(resp), status_code=200)
    except Exception as exc:
        logger.error("Heatmap failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.on_event("startup")
async def on_startup():
    # 自动清洗 Excel 并加载先验
    try:
        subprocess.run(["python", "excel_cleaner_and_formatter.py"], check=True)
        p = Path("output/cleaned_data.json")
        global priors
        if p.exists():
            priors = json.loads(p.read_text(encoding="utf-8"))
        else:
            priors = {}
        brain.priors = priors
        logger.info("Loaded priors from cleaned_data.json")
    except Exception as e:
        logger.warning("Could not load priors on startup: %s", e)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutdown complete")


def run_api() -> None:
    port = int(os.getenv("PORT", "10000"))
    # Clean cached bytecode and restart any lingering uvicorn instances
    subprocess.run("find . -name '*.pyc' -delete", shell=True, check=False)
    subprocess.run(["pkill", "-f", "uvicorn"], check=False)
    logger.info("Starting API on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run_api()

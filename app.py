import asyncio
import atexit
import base64
import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

# fmt: off
import analyzer
import brain
from analyzer import (compute_position_probabilities,
                      fuse_predictions_with_heatmap, fuse_score_matrices,
                      predict_scratch_card, probability_heatmap,
                      render_heatmap)
from env_config import EnvConfig
from strategy_types import Strategy

# fmt: on
brain.priors_map: Dict[str, Dict[int, float]] = {}

# —— Logging setup —————————————————————————————————————————————————————————————
log_handlers = [
    logging.StreamHandler(),
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

# —— Priors config ———————————————————————————————————————————————————————————
PRIORS_DIR = Path("assets")
_IO_POOL = ThreadPoolExecutor(max_workers=1)
atexit.register(_IO_POOL.shutdown)

_DIM_RE = re.compile(r"^(\d+)x(\d+)$")


def _load_priors_files() -> Dict[str, Dict[int, float]]:
    priors: Dict[str, Dict[int, float]] = {}
    for path in PRIORS_DIR.glob("priors_*.json"):
        try:
            priors[path.stem.replace("priors_", "")] = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover - corrupted JSON
            logger.error("Failed to load %s: %s", path, exc)
            # 中文說明：讀取 priors_*.json 檔案失敗，通常是 JSON 格式錯誤
    if not priors:
        priors["10x12"] = compute_position_probabilities("samples", 10, 12)
    return priors


async def load_priors_async() -> Dict[str, Dict[int, float]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_IO_POOL, _load_priors_files)


def find_closest_prior_key(shape: Tuple[int, int], candidates: Dict[str, Any]) -> str:
    rows, cols = shape

    def score(key: str) -> Tuple[int, float]:
        m = _DIM_RE.match(key)
        if not m:
            return (float("inf"), float("inf"))
        kr, kc = int(m.group(1)), int(m.group(2))
        cells_diff = abs((kr * kc) - (rows * cols))
        aspect_diff = abs((kr / kc) - (rows / cols))
        return (cells_diff, aspect_diff)

    return min(candidates, key=score)


def get_prior_for_shape(rows: int, cols: int) -> Dict[int, float]:
    priors_map = brain.priors_map or {}
    if not priors_map:
        return {}
    key = find_closest_prior_key((rows, cols), priors_map)
    return priors_map.get(key, {})


async def warm_up() -> None:
    logging.info("[warm-up] Loading priors…")
    priors = await load_priors_async()
    brain.priors_map.clear()
    brain.priors_map.update(priors)
    logging.info(f"[warm-up] Loaded {len(priors)} prior shapes")

    analyzer.load_all_global_pos_freqs(str(analyzer.DEFAULT_NPZ_DIR))

    analyzer.load_all_sample_stats("samples")
    shapes = analyzer.list_loaded_sample_shapes()
    logging.info("★★ Samples 已加载 shapes: %s", shapes)
    if (4, 5) not in shapes:
        logging.error("样本 4×5 未加载成功！")


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
def _warm_up() -> None:                  ### 新增 ↓
    analyzer.load_all_global_pos_freqs("out_npz")   # 全局熱力圖
    analyzer.load_all_sample_stats("samples")       # 較小的樣本統計
    print("✅ Warm-up 完成")                         ### 可選 log

@app.on_event("startup")                 ### 新增 ↓
async def _startup() -> None:
    threading.Thread(target=_warm_up, daemon=True).start()

@app.get("/debug/priors", response_class=JSONResponse)
async def debug_priors():
    return brain.priors_map


@app.get("/debug/number_distribution", response_class=JSONResponse)
async def debug_number_distribution(
    rows: int, cols: int, mode: Optional[str] = None
) -> Dict[str, Dict[str, int]]:
    """Return per-number position counts for the given board size."""
    dist = analyzer.compute_number_distribution("samples", rows, cols, mode=mode)
    result = {
        str(n): {f"{r + 1},{c + 1}": int(cnt) for (r, c), cnt in pos.items()}
        for n, pos in dist.items()
    }
    return result


@app.get("/debug/global_freqs", response_class=JSONResponse)
async def debug_global_freqs() -> List[str]:
    """Return board sizes with available global frequency files."""
    shapes: set[str] = set()
    for p in analyzer.DEFAULT_NPZ_DIR.glob("global_pos_freq_*x*.npz"):
        m = re.search(r"_(\d+x\d+)\.npz$", p.name)
        if m:
            shapes.add(m.group(1))
    return sorted(shapes)


@app.get("/health/samples")
def health_samples():
    shapes = analyzer.list_loaded_sample_shapes()
    return {"loaded_sample_shapes": shapes}

def _warm_up():
    # 1. 先載現有熱力圖
    analyzer.load_all_global_pos_freqs("out_npz")
    # 2. 若缺 shape，就異步補跑
    if not pathlib.Path("out_npz").glob("global_pos_freq_*.npz"):
        subprocess.run(
            ["python", "build_global_pos_freq.py", "-s", "samples", "-o", "out_npz"],
            check=True
        )
        analyzer.load_all_global_pos_freqs("out_npz")  # 重新註冊
    # 3. 樣本統計
    analyzer.load_all_sample_stats("samples")
    print("✅ Warm-up done.")

@app.on_event("startup")
async def startup_event() -> None:
    await warm_up()
    # 这里 **不要** 再调用 load_all_sample_stats 或其他会重复加载的函数


# ==== Schemas ==============================================================
class GridRequest(BaseModel):
    grid: List[List[Union[int, str]]]
    target_num: Optional[int] = None
    iterations: Optional[int] = None
    global_iter: Optional[int] = None
    focus_iter: Optional[int] = None
    top_n: Optional[int] = None
    epsilon: Optional[float] = None
    result_top_k: Optional[int] = None
    sample_gamma: Optional[float] = 0.9
    use_neighbor_lock: bool = True
    fusion_alpha: Optional[float] = None
    pseudo_count: Optional[float] = None
    exclude_filled: bool = True
    strategy: Strategy = Strategy.LEGACY


class Prediction(BaseModel):
    row: int
    col: int
    candidates: List[int]
    probability: float  # percentage
    reasons: List[str]
    module_scores: Dict[str, float]


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    top_predictions: List[Prediction]
    top_recommendations: Optional[List[Prediction]] = None
    full_probabilities: Dict[str, Dict[str, float]]
    final_recommendations: List[Dict[str, Any]]


class HeatmapRequest(BaseModel):
    grid: List[List[Union[int, str]]]
    k: Optional[int] = None
    target_num: Optional[int] = None
    iterations: int = 1000
    seed: int = 0
    output_format: Literal["base64", "raw", "json"] = "base64"
    sample_gamma: Optional[float] = 0.9
    use_neighbor_lock: bool = True
    fusion_alpha: Optional[float] = None


class HeatmapResponse(BaseModel):
    prob_map: Union[List[List[float]], Dict[str, List[List[float]]]]
    heatmap: Optional[str] = None
    predictions: Optional[List[Prediction]] = None
    top_predictions: Optional[List[Prediction]] = None
    top_recommendations: Optional[List[Prediction]] = None
    full_probabilities: Optional[Dict[str, Dict[str, float]]] = None
    final_recommendations: Optional[List[Dict[str, Any]]] = None


class FusionRequest(BaseModel):
    predict_scores: List[List[float]]
    heatmap_prob_map: List[List[float]]
    alpha: Optional[float] = 0.5
    top_n: int = 5
    target_num: Optional[int] = None


class FusionResult(BaseModel):
    row: int
    col: int
    final_score: float


# ==== Startup & ENV parsing =================================================
startup_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# 默认环境变量
os.environ.setdefault("PHASE1_ITERATIONS", "5000")
os.environ.setdefault("PHASE2_ITERATIONS", "1000")
os.environ.setdefault("PHASE2_TOP_N", "10")
os.environ.setdefault("PHASE2_EPSILON", "0.05")
os.environ.setdefault("LOG_LEVEL", "INFO")

env = EnvConfig()


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


def _to_1_based(preds: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Convert row/col fields in predictions to 1-based indexing."""
    if not preds:
        return []
    result = []
    for p in preds:
        p = p.copy()
        if "row" in p:
            p["row"] = int(p["row"]) + 1
        if "col" in p:
            p["col"] = int(p["col"]) + 1
        result.append(p)
    return result


def _full_probs_to_1_based(
    probs: Dict[Tuple[int, int], Dict[int, float]],
) -> Dict[str, Dict[str, float]]:
    converted: Dict[str, Dict[str, float]] = {}
    for (r, c), pmap in probs.items():
        key = f"{int(r) + 1},{int(c) + 1}"
        inner = {str(int(num)): safe_float(p) * 100 for num, p in pmap.items()}
        converted[key] = inner
    return converted


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
        logger.debug("Loaded priors sizes: %s", list(brain.priors_map.keys()))
        # 格式校驗
        if not req.grid or not all(isinstance(row, list) for row in req.grid):
            raise ValueError("Invalid grid format: expected List[List[int]].")
        rows, cols = len(req.grid), len(req.grid[0])
        if rows < 2 or cols < 2:
            raise ValueError("Grid must be at least 2x2")
        max_val = rows * cols
        known = []
        for row in req.grid:
            for v in row:
                if v in (-1, 0, ""):
                    continue
                known.append(int(v))
        if len(known) != len(set(known)):
            raise ValueError("Grid contains duplicate numbers")
        if any(v < 1 or v > max_val for v in known):
            raise ValueError(f"Numbers must be between 1 and {max_val}")
        if req.target_num is not None and not (1 <= req.target_num <= max_val):
            raise HTTPException(
                status_code=400,
                detail=f"target_num must be between 1 and {max_val}",
            )

        key = f"{rows}x{cols}"
        priors = brain.priors_map.get(key)
        if priors is None:
            logger.warning("No priors for %s, computing on-the-fly", key)
            # 中文說明：指定尺寸缺少先驗分布，將即時計算，可能稍慢
            priors = compute_position_probabilities("samples", rows, cols)
            brain.priors_map[key] = priors

        # 参数
        phase1 = req.iterations or env.phase1_iter
        phase2 = req.focus_iter or env.phase2_iter
        top_n = req.top_n or env.phase2_top_n
        eps = req.epsilon or env.phase2_epsilon
        top_k = req.result_top_k or env.result_top_k

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
        # 中文說明：預測啟動，記錄使用的盤面大小、目標數字與主要參數（ph1 為全局模擬次數，ph2 為焦點模擬次數）

        # 调用核心推理
        grid_np = np.asarray(req.grid, dtype=object)
        # 正規化空格：0 / "" 皆視為 -1
        is_blank = (grid_np == -1) | (grid_np == 0) | (grid_np == "")
        grid_norm = np.where(is_blank, -1, grid_np).astype(int).tolist()
        force_legacy = False
        if "strategy" in req.model_fields_set and req.strategy == Strategy.LEGACY:
            force_legacy = True
        if (
            "fusion_alpha" in req.model_fields_set
            and req.fusion_alpha is not None
            and req.fusion_alpha > 0
        ):
            force_legacy = True

        fusion_alpha = req.fusion_alpha if req.fusion_alpha is not None else 0.1
        result = predict_scratch_card(
            grid=grid_norm,
            target_num=req.target_num,
            iterations=phase1,
            global_iter=req.global_iter,
            focus_iter=phase2,
            top_n=top_n,
            epsilon=eps,
            result_top_k=top_k,
            priors=priors,
            sample_gamma=req.sample_gamma,
            use_neighbor_lock=req.use_neighbor_lock,
            fusion_alpha=fusion_alpha,
            force_legacy=force_legacy,
            pseudo_count=req.pseudo_count or 0.0,
            strategy=req.strategy,
        )

        if req.target_num is not None:
            hm_iter = phase1 if os.getenv("FAST_TEST") != "1" else min(phase1, 100)
            heat = probability_heatmap(
                grid_norm,
                req.target_num,
                hm_iter,
                sample_gamma=req.sample_gamma,
                history_dir="samples",
            )
            fusion_alpha = req.fusion_alpha if req.fusion_alpha is not None else 0.7
            result["final_recommendations"] = fuse_predictions_with_heatmap(
                heat,
                result.get("top_predictions", []),
                fusion_alpha=fusion_alpha,
                top_k=top_k,
            )

        full_probs = result.get("full_probabilities", {})
        clean_probs = _full_probs_to_1_based(full_probs)

        preds = _to_1_based(result.get("predictions"))
        tops = _to_1_based(result.get("top_predictions"))
        recs = _to_1_based(result.get("final_recommendations"))
        top_recs = _to_1_based(result.get("top_recommendations"))

        payload = {
            "predictions": preds,
            "top_predictions": tops,
            "full_probabilities": clean_probs,
            "sample_gamma_used": req.sample_gamma,
            "final_recommendations": recs,
            "top_recommendations": top_recs,
            "strategy": result.get("strategy"),
        }
        safe_payload = sanitize_floats(payload)
        logger.info("✅ Response ready")
        # 中文說明：主要預測流程完成，回傳結果前的確認訊息
        return JSONResponse(content=safe_payload, status_code=200)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction failed", exc_info=True)
        # 中文說明：預測過程發生未預期錯誤，附帶 traceback 供偵錯
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
        grid_np = np.asarray(req.grid, dtype=object)
        is_blank = (grid_np == -1) | (grid_np == 0) | (grid_np == "")
        grid_norm = np.where(is_blank, -1, grid_np).astype(int).tolist()

        prob = probability_heatmap(
            grid_norm,
            k_eff,
            iters,
            seed=req.seed,
            sample_gamma=req.sample_gamma,
            history_dir="samples",
        )

        rows, cols = len(req.grid), len(req.grid[0])
        key = f"{rows}x{cols}"
        priors = brain.priors_map.get(key)
        if priors is None:
            logger.warning("No priors for %s, computing on-the-fly", key)
            # 中文說明：缺少對應尺寸先驗，需即時計算才能繼續
            priors = compute_position_probabilities("samples", rows, cols)
            brain.priors_map[key] = priors

        pred_result = predict_scratch_card(
            grid=grid_norm,
            target_num=req.target_num,
            iterations=env.phase1_iter,
            global_iter=None,
            focus_iter=env.phase2_iter,
            top_n=env.phase2_top_n,
            epsilon=env.phase2_epsilon,
            result_top_k=3,
            priors=priors,
            sample_gamma=req.sample_gamma,
            use_neighbor_lock=req.use_neighbor_lock,
        )

        fusion_alpha = req.fusion_alpha or 0.7
        if isinstance(prob, np.ndarray) and req.target_num is not None:
            pred_result["final_recommendations"] = fuse_predictions_with_heatmap(
                prob,
                pred_result.get("top_predictions", []),
                fusion_alpha=fusion_alpha,
                top_k=3,
            )

        full_probs = pred_result.get("full_probabilities", {})
        clean_probs = _full_probs_to_1_based(full_probs)

        preds = _to_1_based(pred_result.get("predictions"))
        tops = _to_1_based(pred_result.get("top_predictions"))
        recs = _to_1_based(pred_result.get("final_recommendations"))
        top_recs = _to_1_based(pred_result.get("top_recommendations"))

        if isinstance(prob, dict):
            pm = {str(int(k)): v.tolist() for k, v in prob.items()}
            resp = {
                "prob_map": pm,
                "heatmap": None,
                "predictions": preds,
                "top_predictions": tops,
                "full_probabilities": clean_probs,
                "final_recommendations": recs,
                "top_recommendations": top_recs,
                "strategy": pred_result.get("strategy"),
            }
        elif req.output_format.lower() in ("raw", "json"):
            resp = {
                "prob_map": prob.tolist(),
                "heatmap": None,
                "predictions": preds,
                "top_predictions": tops,
                "full_probabilities": clean_probs,
                "final_recommendations": recs,
                "top_recommendations": top_recs,
                "sample_gamma_used": req.sample_gamma,
                "strategy": pred_result.get("strategy"),
            }
        else:
            img = render_heatmap(prob, req.output_format)
            b64 = base64.b64encode(img).decode() if isinstance(img, bytes) else img
            resp = {
                "prob_map": prob.tolist(),
                "heatmap": b64,
                "predictions": preds,
                "top_predictions": tops,
                "full_probabilities": clean_probs,
                "final_recommendations": recs,
                "top_recommendations": top_recs,
                "strategy": pred_result.get("strategy"),
            }

        return JSONResponse(content=sanitize_floats(resp), status_code=200)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Heatmap failed", exc_info=True)
        # 中文說明：產生熱力圖時發生錯誤，紀錄原因與堆疊
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/fuse",
    response_model=List[FusionResult],
    response_class=JSONResponse,
    status_code=200,
)
async def fuse(req: FusionRequest):
    try:
        pred_arr = np.asarray(req.predict_scores, dtype=float)
        heat_arr = np.asarray(req.heatmap_prob_map, dtype=float)
        if pred_arr.shape != heat_arr.shape:
            raise HTTPException(
                status_code=400,
                detail="predict_scores and heatmap_prob_map must have the same shape",
            )
        alpha = req.alpha if req.alpha is not None else 0.5
        fused = fuse_score_matrices(
            pred_arr,
            heat_arr,
            fusion_alpha=alpha,
            top_k=req.top_n,
        )
        recs = _to_1_based(fused)
        return JSONResponse(content=sanitize_floats(recs), status_code=200)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Fusion failed", exc_info=True)
        # 中文說明：合併兩組分數矩陣時發生例外，印出錯誤細節
        raise HTTPException(status_code=500, detail=str(exc))


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutdown complete")
    # 中文說明：FastAPI 伺服器已關閉，用於觀察服務停止時間點


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",  # 或者你的应用模块路径
        host="0.0.0.0",
        port=port,
        log_level="info",
    )

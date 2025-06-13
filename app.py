# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import pandas as pd
import json
import os
import logging
import asyncio
import glob
from typing import Dict, List, Optional, Tuple, Any
from brain import process_single_board, process_batch, load_grid_from_file
from analyzer import analyze_board
from pydantic import BaseModel, Field, validator, ConfigDict
from functools import lru_cache
from joblib import Parallel, delayed

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s:%(name)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- 路徑常數 & 輔助 ---------- #
import os, glob, zipfile, json, logging
from typing import Dict, Any, List, Tuple

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.getenv("DATA_DIR") or os.path.join(BASE_DIR, "samples", "data")
EXTRACT_DIR = os.path.join(DATA_DIR, "temp_extract")
os.makedirs(EXTRACT_DIR, exist_ok=True)
log = logging.getLogger(__name__)

def _extract_all_zips() -> None:
    """解 DATA_DIR/*.zip → EXTRACT_DIR（重複執行不重複解）。"""
    for zp in glob.glob(os.path.join(DATA_DIR, "*.zip")):
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(EXTRACT_DIR)
            log.info(f"✅ Extracted: {os.path.basename(zp)}")
        except Exception as e:
            log.warning(f"❌ Failed extracting {zp}: {e}")

def _load_all_json() -> Dict[str, Any]:
    """遞迴讀 DATA_DIR 與 EXTRACT_DIR 內所有 .json 檔。"""
    patterns = [
        os.path.join(DATA_DIR, "*.json"),
        os.path.join(EXTRACT_DIR, "**", "*.json"),
    ]
    paths: List[str] = []
    for p in patterns:
        paths.extend(glob.glob(p, recursive=True))

    objs: Dict[str, Any] = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as f:
                objs[name] = json.load(f)
            log.info(f"🟢 Loaded JSON: {name}")
        except Exception as e:
            log.warning(f"🔴 Bad JSON {path}: {e}")
    return objs

def load_data_resources() -> Tuple[List[Dict], Dict[str, Any]]:
    """回傳 (knowledge_base, json_objs)。KB 檔不存在就給空 list。"""
    kb_path = os.path.join(DATA_DIR, "math_algo_kb.json")
    kb: List[Dict] = []
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                kb = json.load(f).get("concepts", [])
            log.info(f"🟢 KB loaded: {len(kb)} concepts")
        except Exception as e:
            log.warning(f"🔴 Failed loading KB: {e}")

    _extract_all_zips()
    json_objs = _load_all_json()
    log.info(f"🧮 Total JSON files = {len(json_objs)}")
    return kb, json_objs
# ---------- 路徑常數 & 輔助 END ---------- #

app = FastAPI(
    title="Scratch Card Analysis API",
    version="1.0.0",
    description="Scratch card grid analysis service providing top-3 predictions.",
    openapi_version="3.1.0"
)

# 初始化 knowledge base 和 heatmaps
math_algo_kb, json_objs = load_data_resources()
heatmaps = {k: v for k, v in json_objs.items() if "heatmap" in k.lower()}

class AnalysisRequest(BaseModel):
    grid: List[List[float]] = Field(..., description="2D array, -1 for hidden cells")
    weights: Optional[Dict[str, float]] = None
    mode: str = Field("predict", description="Analysis mode: 'predict' or 'heatmap'")
    target_num: Optional[int] = Field(None, description="Target number to predict")
    json_heatmap: str = Field("samples/data/json", description="JSON heatmap folder")
    model_path: str = Field("models/model.pkl", description="Trained model path")
    global_heatmap_path: Optional[str] = Field(None, description="Global heatmap JSON file path")

    model_config = ConfigDict(protected_namespaces=())

    @validator("grid")
    def validate_grid(cls, grid):
        grid_array = np.atleast_2d(np.array(grid, dtype=np.int64))
        if grid_array.ndim != 2 or grid_array.shape[0] < 4 or grid_array.shape[1] < 4 or \
           grid_array.shape[0] > 20 or grid_array.shape[1] > 20:
            raise ValueError("Grid size must be 4x4 to 20x20")
        if not np.any(grid_array == -1):
            raise ValueError("Grid must contain at least one hidden cell (-1) for prediction")
        open_nums = grid_array[grid_array != -1]
        if len(open_nums) > 0 and (len(set(open_nums)) != len(open_nums) or max(open_nums) > grid_array.size or min(open_nums) < 1):
            raise ValueError(f"Grid values must be unique and in range 1 to {grid_array.size} or -1")
        return grid_array.tolist()

class Prediction(BaseModel):
    row: int
    col: int
    predicted_digit: int
    confidence: float
    module_scores: Dict[str, float]
    true_digit: Optional[int] = None

    model_config = ConfigDict(protected_namespaces=())

class AnalysisResponse(BaseModel):
    predictions: List[Prediction]
    error: Optional[str]
    source: str = "🔥 from real API"
    reasoning: List[str]

    model_config = ConfigDict(protected_namespaces=())

DEFAULT_WEIGHTS = {
    "compute_dynamic_hot_cold_vectorized": 0.15,
    "compute_dynamic_hot_cold_advanced": 0.2,
    "compute_block_heatmap_vectorized": 0.1,
    "idw_vectorized": 0.1,
    "compute_global_diff_heatmap": 0.05,
    "compute_focus_score": 0.1,
    "detect_skip_patterns": 0.05,
    "compute_difference_trend": 0.05,
    "detect_mirror_sequences": 0.05,
    "connectivity_heatmap": 0.05,
    "sequence_tail_analyzer": 0.05,
    "analyze_number_patterns": 0.05
}

@lru_cache(maxsize=1000)
def cache_board_analysis(
    grid_tuple: Tuple[float, ...], shape: Tuple[int, int], target_num: int, model_path: str
) -> Tuple[List[Dict], List[str]]:
    try:
        grid = np.array(grid_tuple, dtype=np.int64).reshape(shape)
        if grid.ndim != 2 or grid.size != shape[0] * shape[1]:
            raise ValueError(f"Invalid grid data for shape {shape}")
        logger.debug(f"Cache hit for grid shape {shape} with target {target_num}")
        predictions, reasoning = perform_board_analysis(grid, target_num, model_path)
        return predictions, reasoning
    except Exception as e:
        logger.error(f"Cache analysis failed: {str(e)}")
        return [], []

def perform_board_analysis(grid: np.ndarray, target_num: int, model_path: str) -> Tuple[List[Dict], List[str]]:
    M, N = grid.shape
    predictions = []
    logger.info(f"Analyzing grid of size {M}x{N} for target number {target_num}")
    
    try:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            raise ValueError(f"Invalid grid type or shape: {type(grid)}, {grid.shape if hasattr(grid, 'shape') else 'None'}")
        if grid.dtype != np.int64:
            grid = grid.astype(np.int64)
            logger.info("Grid cast to int64 for consistency")
        
        empty_yx = np.argwhere(grid == -1)
        if len(empty_yx) == 0:
            raise ValueError("No hidden cells (-1) found for prediction")

        final_score, pred_array, top3, metrics, reasoning = analyze_board(
            grid, DEFAULT_WEIGHTS, True, target_num, None, math_algo_kb, heatmaps, model_path
        )
        heatmap = final_score if final_score.ndim == 2 else np.zeros_like(grid, dtype=float)
        
        predictions.extend([
            {
                "row": int(p["row"]),
                "col": int(p["col"]),
                "predicted_digit": int(p["predicted_digit"]),
                "confidence": float(p["confidence"]),
                "module_scores": {**p["module_scores"], "heatmap": float(heatmap[p["row"], p["col"]])},
                "true_digit": None
            }
            for p in top3
        ])
        
    except Exception as e:
        logger.error(f"Analysis failed for grid: {str(e)}")
        raise
    
    reasoning = [
        f"Remaining numbers: {list(set(range(1, M * N + 1)) - set(grid[grid != -1].flatten()))}",
        f"Target number {target_num} analyzed across {len(predictions)} candidates"
    ]
    logger.info(f"Analysis completed with {len(predictions)} predictions")
    return predictions, reasoning

@app.get("/health")
async def health_check() -> Dict[str, str]:
    logger.info("Health check requested")
    return {"status": "ok"}

@app.post(
    "/predict",
    response_model=AnalysisResponse,
    openapi_extra={"operationId": "predictFromJson"}
)
async def predict(payload: AnalysisRequest) -> JSONResponse:
    logger.info(f"🔍 RAW grid payload = {json.dumps(payload.grid)}")
    
    grid = np.array(payload.grid, dtype=np.int64)
    logger.info(f"🔍 AFTER reshape arr.shape = {grid.shape}")
    
    if grid.ndim != 2 or grid.shape[0] < 4 or grid.shape[1] < 4 or grid.shape[0] > 20 or grid.shape[1] > 20:
        raise HTTPException(status_code=422, detail="Grid must be a 4x4 to 20x20 2D numeric matrix")
    
    flat = grid[grid != -1].flatten()
    if len(flat) != len(set(flat)):
        raise HTTPException(status_code=422, detail="Grid values except -1 must be unique and non-repeating")
    
    target = 6 if payload.mode == "predict" and payload.target_num is None else payload.target_num
    if target is None:
        logger.warning("No target_num specified, defaulting to 6")
    
    try:
        final_score, predictions, top3, metrics, reasoning = analyze_board(
            grid, payload.weights or DEFAULT_WEIGHTS, True, target, payload.json_heatmap,
            math_algo_kb, heatmaps, payload.model_path, payload.global_heatmap_path
        )
        heatmap = final_score if final_score.ndim == 2 else np.zeros_like(grid, dtype=float)
        
        result = AnalysisResponse(
            predictions=[
                Prediction(
                    row=int(p["row"]),
                    col=int(p["col"]),
                    predicted_digit=int(p["predicted_digit"]),
                    confidence=float(p["confidence"]),
                    module_scores={**p["module_scores"], "heatmap": float(heatmap[p["row"], p["col"]])}
                )
                for p in top3
            ],
            error=None,
            source="🔥 from real API",
            reasoning=reasoning
        )
        return JSONResponse(
            status_code=200,
            content=result.dict()
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        error_resp = AnalysisResponse(predictions=[], error=str(e), source="🔥 from real API", reasoning=[])
        return JSONResponse(status_code=500, content=error_resp.dict())

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> JSONResponse:
    logger.info(f"Received upload request for file: {file.filename}")
    try:
        if not file.filename.endswith(('.json', '.csv', '.xls', '.xlsx')):
            error_msg = f"Unsupported file format: {file.filename}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        input_path = os.path.join("samples", "data", file.filename)
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Successfully saved uploaded file to {input_path}")
        
        output_prefix = os.path.join("samples", "output", os.path.splitext(file.filename)[0])
        weights = DEFAULT_WEIGHTS
        json_heatmap = os.path.join("samples", "data", "json")
        
        background_tasks.add_task(
            process_single_board, input_path, weights, True, output_prefix, None, json_heatmap
        )
        logger.info(f"Scheduled background processing for {input_path}")
        
        return JSONResponse(
            content={"message": f"File {file.filename} uploaded, processing started", "output_path": output_prefix},
            status_code=200
        )
    
    except HTTPException as e:
        logger.error(f"File upload failed: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_process(
    input_folder: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> JSONResponse:
    logger.info(f"Received batch processing request for folder: {input_folder}")
    try:
        if not os.path.exists(input_folder):
            error_msg = f"Folder {input_folder} does not exist"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        files = [f for f in os.listdir(input_folder) if f.endswith(('.json', '.csv', '.xls', '.xlsx'))]
        logger.info(f"Found {len(files)} valid files in {input_folder}: {files}")
        
        output_folder = os.path.join("samples", "output", f"batch_{os.path.basename(input_folder)}")
        weights = DEFAULT_WEIGHTS
        json_heatmap = os.path.join("samples", "data", "json")
        
        background_tasks.add_task(
            process_batch, input_folder, weights, True, output_folder, None, json_heatmap
        )
        logger.info(f"Scheduled batch processing, results will be saved to {output_folder}")
        
        return JSONResponse(
            content={"message": f"Batch processing started with {len(files)} files, results will be saved to {output_folder}"},
            status_code=200
        )
    
    except HTTPException as e:
        logger.error(f"Batch processing failed: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def save_results_to_file(
    scores: np.ndarray,
    predictions: np.ndarray,
    best_pos: List[Tuple[int, int, float, Dict[str, float]]],
    output_filepath: str,
    output_format: str
) -> None:
    from brain import save_results_to_file as brain_save
    logger.info(f"Saving results to {output_filepath} in {output_format} format")
    try:
        brain_save(scores, predictions, best_pos, output_filepath, output_format)
        logger.info(f"Successfully saved results to {output_filepath}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_filepath}: {str(e)}")
        raise

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, full_path: str) -> JSONResponse:
    logger.debug(f"Catch-all: {request.method} {full_path}")
    return JSONResponse(status_code=200, content={"status": "running"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
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
from pydantic import BaseModel, Field, validator
from functools import lru_cache

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

app = FastAPI(
    title="Scratch Card Analysis API",
    version="1.0.0",
    description="Scratch card grid analysis service providing top-3 predictions.",
    openapi_version="3.1.0"
)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "samples", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def load_data_resources() -> Tuple[List[Dict], Dict[str, Any]]:
    kb_path = os.path.join(DATA_DIR, "math_algo_kb.json")
    heatmap_paths = glob.glob(os.path.join(DATA_DIR, "*_heatmap.json"))
    
    default_kb = [
        {"concept": "basic_arithmetic", "description": "Basic addition and subtraction rules", "weight": 0.5},
        {"concept": "pattern_recognition", "description": "Detecting sequences and patterns", "weight": 0.5}
    ]
    math_algo_kb: List[Dict] = []
    heatmaps: Dict[str, Any] = {}
    
    if os.path.exists(kb_path):
        try:
            with open(kb_path, 'r', encoding="utf-8") as f:
                math_algo_kb = json.load(f)["concepts"]
        except (OSError, json.JSONDecodeError, KeyError):
            math_algo_kb = default_kb
    else:
        math_algo_kb = default_kb
    
    for hp in heatmap_paths:
        name = os.path.splitext(os.path.basename(hp))[0]
        try:
            with open(hp, 'r', encoding="utf-8") as f:
                heatmaps[name] = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    
    return math_algo_kb, heatmaps

math_algo_kb, heatmaps = load_data_resources()

class AnalysisRequest(BaseModel):
    grid: List[List[float]] = Field(..., description="2D array, -1 for hidden cells")
    weights: Optional[Dict[str, float]] = None
    mode: str = Field("predict", description="Analysis mode: 'predict' or 'heatmap'")
    target_num: Optional[int] = Field(None, description="Target number to predict")
    json_heatmap: str = Field("samples/data/json", description="JSON heatmap folder")
    model_path: str = Field("models/model.pkl", description="Trained model path")

    @validator("grid")
    def validate_grid(cls, grid):
        grid_array = np.atleast_2d(np.array(grid, dtype=float))
        if grid_array.ndim != 2 or grid_array.shape[0] < 4 or grid_array.shape[1] < 4 or \
           grid_array.shape[0] > 20 or grid_array.shape[1] > 20:
            raise ValueError
        if not np.any(grid_array == -1):
            raise ValueError
        open_nums = grid_array[grid_array != -1]
        if len(open_nums) > 0 and (len(set(open_nums)) != len(open_nums) or max(open_nums) > grid_array.size or min(open_nums) < 1):
            raise ValueError
        return grid_array.tolist()

class Prediction(BaseModel):
    row: int
    col: int
    predicted_digit: int
    confidence: float
    module_scores: Dict[str, float]
    true_digit: Optional[int] = None

class AnalysisResponse(BaseModel):
    predictions: List[Prediction]
    error: Optional[str]
    source: str = "🔥 from real API"
    reasoning: List[str]

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
def cache_board_analysis(grid_tuple: Tuple[float, ...], shape: Tuple[int, int], target_num: int, model_path: str) -> Tuple[List[Dict], List[str]]:
    try:
        grid = np.array(grid_tuple).reshape(shape)
        if grid.ndim != 2 or grid.size != shape[0] * shape[1]:
            raise ValueError
        predictions, reasoning = perform_board_analysis(grid, target_num, model_path)
        return predictions, reasoning
    except Exception:
        return [], []

def perform_board_analysis(grid: np.ndarray, target_num: int, model_path: str) -> Tuple[List[Dict], List[str]]:
    M, N = grid.shape
    predictions = []
    
    try:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            raise ValueError
        
        empty_yx = np.argwhere(grid == -1)
        if len(empty_yx) == 0:
            raise ValueError

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
    except Exception:
        raise
    
    reasoning = [
        f"Remaining numbers: {list(set(range(1, M * N + 1)) - set(grid[grid != -1].flatten()))}",
        f"Target number {target_num} analyzed across {len(predictions)} candidates"
    ]
    return predictions, reasoning

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=AnalysisResponse)
async def predict(payload: AnalysisRequest) -> JSONResponse:
    grid = np.array(payload.grid, dtype=float)
    
    if grid.ndim != 2 or grid.shape[0] < 4 or grid.shape[1] < 4 or grid.shape[0] > 20 or grid.shape[1] > 20:
        raise HTTPException(422, "Grid must be a 4x4 to 20x20 2D numeric matrix")
    
    flat = grid[grid != -1].flatten()
    if len(flat) != len(set(flat)):
        raise HTTPException(422, "Grid values except -1 must be unique and non-repeating")
    
    target = 6 if payload.mode == "predict" and payload.target_num is None else payload.target_num
    if target is None:
        pass
    
    try:
        final_score, predictions, top3, metrics, reasoning = analyze_board(
            grid, payload.weights or DEFAULT_WEIGHTS, True, target, payload.json_heatmap,
            math_algo_kb, heatmaps, payload.model_path
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
    except Exception:
        error_resp = AnalysisResponse(predictions=[], error="Prediction failed", source="🔥 from real API", reasoning=[])
        return JSONResponse(status_code=500, content=error_resp.dict())

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()) -> JSONResponse:
    try:
        if not file.filename.endswith(('.json', '.csv', '.xls', '.xlsx')):
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        input_path = os.path.join("samples", "data", file.filename)
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        output_prefix = os.path.join("samples", "output", os.path.splitext(file.filename)[0])
        weights = DEFAULT_WEIGHTS
        json_heatmap = os.path.join("samples", "data", "json")
        
        background_tasks.add_task(
            process_single_board, input_path, weights, True, output_prefix, None, json_heatmap
        )
        
        return JSONResponse(
            content={"message": f"File {file.filename} uploaded, processing started", "output_path": output_prefix},
            status_code=200
        )
    except HTTPException as e:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="File upload failed")

@app.post("/batch")
async def batch_process(input_folder: str = Form(...), background_tasks: BackgroundTasks = BackgroundTasks()) -> JSONResponse:
    try:
        if not os.path.exists(input_folder):
            raise HTTPException(status_code=404, detail="Folder does not exist")
        
        files = [f for f in os.listdir(input_folder) if f.endswith(('.json', '.csv', '.xls', '.xlsx'))]
        
        output_folder = os.path.join("samples", "output", f"batch_{os.path.basename(input_folder)}")
        weights = DEFAULT_WEIGHTS
        json_heatmap = os.path.join("samples", "data", "json")
        
        background_tasks.add_task(
            process_batch, input_folder, weights, True, output_folder, None, json_heatmap
        )
        
        return JSONResponse(
            content={"message": f"Batch processing started with {len(files)} files, results will be saved to {output_folder}"},
            status_code=200
        )
    except HTTPException as e:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Batch processing failed")

def save_results_to_file(scores: np.ndarray, predictions: np.ndarray, best_pos: List[Tuple[int, int, float, Dict[str, float]]], output_filepath: str, output_format: str) -> None:
    from brain import save_results_to_file as brain_save
    try:
        brain_save(scores, predictions, best_pos, output_filepath, output_format)
    except Exception:
        raise

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, full_path: str) -> JSONResponse:
    return JSONResponse(status_code=200, content={"status": "running"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
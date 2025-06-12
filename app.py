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
from modules import ScratchSolver  # 確保從 modules.py 導入

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
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

# Data directory setup
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "samples", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Load knowledge base and heatmaps
def load_data_resources() -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Load knowledge base and heatmaps from data directory with detailed logging.

    Returns:
        Tuple[List[Dict], Dict[str, Any]]: Knowledge base and heatmaps.
    """
    kb_path = os.path.join(DATA_DIR, "math_algo_kb.json")
    heatmap_paths = glob.glob(os.path.join(DATA_DIR, "*_heatmap.json"))
    
    math_algo_kb: List[Dict] = []
    heatmaps: Dict[str, Any] = {}
    
    if os.path.exists(kb_path):
        try:
            with open(kb_path, 'r', encoding="utf-8") as f:
                math_algo_kb = json.load(f)["concepts"]
            logger.info(f"Successfully loaded knowledge base from {kb_path} with {len(math_algo_kb)} concepts")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load knowledge base from {kb_path}: {str(e)}")
    else:
        logger.warning(f"Knowledge base file not found at {kb_path}, using empty KB")
    
    for hp in heatmap_paths:
        name = os.path.splitext(os.path.basename(hp))[0]
        try:
            with open(hp, 'r', encoding="utf-8") as f:
                heatmaps[name] = json.load(f)
            logger.info(f"Successfully loaded heatmap {name} from {hp}")
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load heatmap {name} from {hp}: {str(e)}")
    
    if not heatmaps:
        logger.warning("No valid heatmaps loaded, proceeding with empty heatmap data")
    
    return math_algo_kb, heatmaps

math_algo_kb, heatmaps = load_data_resources()

class AnalysisRequest(BaseModel):
    """
    Schema for JSON payload to analyze a scratch card grid.
    """
    grid: List[List[int]] = Field(..., description="2D array, -1 for hidden cells")
    weights: Optional[Dict[str, float]] = None
    mode: str = Field("predict", description="Analysis mode: 'predict' or 'heatmap'")
    target_num: Optional[int] = Field(None, description="Target number to predict")
    json_heatmap: str = Field("samples/data/json", description="JSON heatmap folder")
    model_path: str = Field("models/model.pkl", description="Trained model path")

    @validator("grid")
    def validate_grid(cls, grid):
        grid_array = np.atleast_2d(np.array(grid, dtype=int))
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
    """
    Schema for individual prediction.
    """
    row: int
    col: int
    predicted_digit: int
    confidence: float
    module_scores: Dict[str, float]
    true_digit: Optional[int] = None

class AnalysisResponse(BaseModel):
    """
    Schema for API response.
    """
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
def cache_board_analysis(
    grid_tuple: Tuple[int, ...], shape: Tuple[int, int], target_num: int, model_path: str
) -> Tuple[List[Dict], List[str]]:
    """
    Cache board analysis results.

    Args:
        grid_tuple (Tuple[int, ...]): Flattened grid as tuple for caching.
        shape (Tuple[int, int]): Original grid shape.
        target_num (int): Target number.
        model_path (str): Model path.

    Returns:
        Tuple[List[Dict], List[str]]: Predictions and reasoning.
    """
    try:
        grid = np.array(grid_tuple).reshape(shape)
        if grid.ndim != 2 or grid.size != shape[0] * shape[1]:
            raise ValueError(f"Invalid grid data for shape {shape}")
        logger.debug(f"Cache hit for grid shape {shape} with target {target_num}")
        predictions, reasoning = perform_board_analysis(grid, target_num, model_path)
        return predictions, reasoning
    except Exception as e:
        logger.error(f"Cache analysis failed: {str(e)}")
        return [], []

def perform_board_analysis(grid: np.ndarray, target_num: int, model_path: str) -> Tuple[List[Dict], List[str]]:
    """
    Perform board analysis with detailed logging and validation.

    Args:
        grid (np.ndarray): 2D board array.
        target_num (int): Target number.
        model_path (str): Model path.

    Returns:
        Tuple[List[Dict], List[str]]: Predictions and reasoning.
    """
    M, N = grid.shape
    predictions = []
    logger.info(f"Analyzing grid of size {M}x{N} for target number {target_num}")
    
    try:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            raise ValueError(f"Invalid grid type or shape: {type(grid)}, {grid.shape if hasattr(grid, 'shape') else 'None'}")
        
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
    """
    Check API health status.
    """
    logger.info("Health check requested")
    return {"status": "ok"}

@app.post(
    "/predict",
    response_model=AnalysisResponse,
    openapi_extra={"operationId": "predictFromJson"}
)
async def predict(payload: AnalysisRequest) -> JSONResponse:
    """
    Predict hidden cells for a target number via JSON payload.
    """
    logger.info(f"🔍 RAW grid payload = {json.dumps(payload.grid)}")
    
    grid = np.array(payload.grid, dtype=int)
    logger.info(f"🔍 AFTER reshape arr.shape = {grid.shape}")
    
    if grid.ndim != 2 or grid.shape[0] < 4 or grid.shape[1] < 4 or grid.shape[0] > 20 or grid.shape[1] > 20:
        raise HTTPException(422, "Grid must be a 4x4 to 20x20 2D array")
    if not np.any(grid == -1):
        raise HTTPException(422, "Grid must contain at least one hidden cell (-1) for prediction")
    
    weights = payload.weights if payload.weights else DEFAULT_WEIGHTS
    target_num = payload.target_num if payload.target_num is not None else None
    model_path = payload.model_path
    
    try:
        grid_tuple = tuple(grid.flatten())
        predictions, reasoning = cache_board_analysis(grid_tuple, grid.shape, target_num, model_path)
        if not predictions:
            raise ValueError("No predictions generated")
        
        result = AnalysisResponse(
            predictions=[Prediction(**p) for p in predictions],
            error=None,
            reasoning=reasoning
        )
        return JSONResponse(content=result.dict())
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return JSONResponse(
            content=AnalysisResponse(
                predictions=[],
                error=str(e),
                reasoning=["Analysis failed"]
            ).dict(),
            status_code=500
        )

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload and process a scratch card file with detailed logging.

    Args:
        file (UploadFile): File containing grid data.
        background_tasks (BackgroundTasks): Background task handler.

    Returns:
        JSONResponse: Upload status and output path.
    """
    output_prefix = os.path.join("outputs", f"upload_{os.path.splitext(file.filename)[0]}")
    try:
        file_path = os.path.join("uploads", file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"Uploaded file saved to {file_path}")
        
        background_tasks.add_task(
            process_single_board,
            file_path,
            DEFAULT_WEIGHTS,
            True,
            output_prefix,
            None,
            "samples/data"
        )
        return JSONResponse({"status": "processing", "output_path": output_prefix})
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_process(
    input_folder: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Initiate batch processing of scratch card files with detailed logging.

    Args:
        input_folder (str): Directory containing input files.
        background_tasks (BackgroundTasks): Background task handler.

    Returns:
        JSONResponse: Batch processing status.
    """
    output_folder = os.path.join("outputs", "batch")
    try:
        if not os.path.exists(input_folder):
            raise ValueError(f"Input folder {input_folder} does not exist")
        background_tasks.add_task(
            process_batch,
            input_folder,
            DEFAULT_WEIGHTS,
            True,
            output_folder,
            None,
            "samples/data"
        )
        return JSONResponse({"status": "processing", "output_folder": output_folder})
    except Exception as e:
        logger.error(f"Batch process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def catch_all(request: Request, full_path: str):
    """
    Catch all undefined routes.

    Args:
        request (Request): HTTP request object.
        full_path (str): Requested path.

    Returns:
        JSONResponse: Running status.
    """
    logger.warning(f"Catch-all route triggered for {request.method} {full_path}")
    return JSONResponse({"status": "running", "message": f"Unknown route {full_path}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
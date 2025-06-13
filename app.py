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

# Global variable to store heatmaps
all_heatmaps: Dict[str, Dict] = {}

# Load knowledge base and heatmaps with default fallback
def load_data_resources() -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Load knowledge base and heatmaps from data directory with detailed logging.
    Returns a default knowledge base if the file is not found.
    """
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
            logger.info(f"Successfully loaded knowledge base from {kb_path} with {len(math_algo_kb)} concepts")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load knowledge base from {kb_path}: {str(e)}")
            math_algo_kb = default_kb
            logger.warning(f"Using default knowledge base due to error: {str(e)}")
    else:
        math_algo_kb = default_kb
        logger.warning(f"Knowledge base file not found at {kb_path}, using default KB with {len(default_kb)} concepts")
    
    for heatmap_path in heatmap_paths:
        name = os.path.splitext(os.path.basename(heatmap_path))[0]
        try:
            with open(heatmap_path, 'r', encoding="utf-8") as f:
                heatmaps[name] = json.load(f)
            logger.info(f"Successfully loaded heatmap {name} from {heatmap_path}")
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load heatmap {name} from {heatmap_path}: {str(e)}")
    
    if not heatmaps:
        logger.warning("No valid heatmaps loaded, proceeding with empty heatmap data")
    
    return math_algo_kb, heatmaps

async def load_all_heatmaps(json_heatmap_dir: str) -> Dict[str, Dict]:
    """
    Load all JSON and ZIP files containing heatmaps from the specified directory.
    
    Parameters:
        json_heatmap_dir (str): Directory containing JSON and ZIP heatmap files.
    
    Returns:
        Dict[str, Dict]: Dictionary of heatmap data with filenames as keys.
    """
    heatmaps = {}
    if not os.path.exists(json_heatmap_dir):
        os.makedirs(json_heatmap_dir, exist_ok=True)
        logger.warning(f"Directory {json_heatmap_dir} created as it did not exist")
        return heatmaps
    
    import zipfile
    for filename in os.listdir(json_heatmap_dir):
        filepath = os.path.join(json_heatmap_dir, filename)
        if filename.endswith('.json'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    heatmaps[filename] = json.load(f)
                logger.info(f"Loaded heatmap from {filename}")
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load JSON heatmap from {filepath}: {e}")
        elif filename.endswith('.zip'):
            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    for zip_info in zip_ref.infolist():
                        if zip_info.filename.endswith('.json'):
                            with zip_ref.open(zip_info.filename) as json_file:
                                heatmaps[f"{filename}_{zip_info.filename}"] = json.load(json_file)
                    logger.info(f"Loaded heatmaps from ZIP {filename}")
            except (zipfile.BadZipFile, json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load ZIP heatmap from {filepath}: {e}")
    
    logger.info(f"Total heatmaps loaded: {len(heatmaps)}")
    return heatmaps

@app.on_event("startup")
async def startup_event():
    """
    Load all heatmaps at application startup.
    """
    global all_heatmaps
    try:
        all_heatmaps = await load_all_heatmaps(os.path.join(DATA_DIR, "json"))
        logger.info(f"Loaded {len(all_heatmaps)} heatmaps during startup")
    except Exception as e:
        logger.error(f"Failed to load heatmaps during startup: {str(e)}")
        all_heatmaps = {}

math_algo_kb, initial_heatmaps = load_data_resources()

class AnalysisRequest(BaseModel):
    """
    Schema for JSON payload to analyze a scratch card grid.
    """
    grid: List[List[float]] = Field(..., description="2D array, -1 for hidden cells")
    weights: Optional[Dict[str, float]] = None
    mode: str = Field("predict", description="Analysis mode: 'predict' or 'heatmap'")
    target_num: Optional[int] = Field(None, description="Target number to predict")
    json_heatmap: str = Field("samples/data/json", description="JSON heatmap folder")
    model_path: str = Field("models/model.pkl", description="Trained model path")

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
    """
    Schema for individual prediction.
    """
    row: int
    col: int
    predicted_digit: int
    confidence: float
    module_scores: Dict[str, float]
    true_digit: Optional[int] = None

    model_config = ConfigDict(protected_namespaces=())

class AnalysisResponse(BaseModel):
    """
    Schema for API response.
    """
    predictions: List[Prediction]
    error: Optional[str]
    source: str = "🔥 from real API"
    reasoning: List[str]

    model_config = ConfigDict(protected_namespaces=())

DEFAULT_WEIGHTS = {
    "compute_dynamic_hot_cold_vectorized": 0.15,  # Corrected key to match modules.py
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
    """
    Cache board analysis results.
    """
    try:
        grid = np.array(grid_tuple, dtype=np.int64).reshape(shape)
        if grid.ndim != 2 or grid.size != shape[0] * shape[1]:
            raise Value
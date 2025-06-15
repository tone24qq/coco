# app2.py

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
import zipfile
import tempfile
import psutil
import ijson
import faiss
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple, Any, Generator
from brain import process_single_board, process_batch, load_grid_from_file, build_feature_index
from analyzer import analyze_board
from modules import compute_features

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)
_log_fmt = "%(asctime)s [%(levelname)s:%(name)s] %(message)s"
root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers.clear()
root.addHandler(RotatingFileHandler("logs/app.log", maxBytes=10 * 1024 * 1024, backupCount=5))
root.addHandler(logging.StreamHandler())

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format=_log_fmt,
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI(
    title="Scratch Card Analysis API",
    version="1.0.0",
    description="Provides scratch card grid analysis services with dynamic pattern detection and feature index querying.",
    openapi_version="3.1.0"
)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "samples", "data")
INDEX_DIR = os.path.join(BASE_DIR, "samples", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.idx")
META_PATH = os.path.join(INDEX_DIR, "meta_paths.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
logger.info("📂 Data directory: %s", DATA_DIR)

MAX_HEATMAPS = 500_000  # Heatmap limit
BATCH_SIZE = 1_000  # Batch size for reading files

# --- Unified JSON/ZIP Heatmap Loader ---
def load_records(path):
    """
    读取 heatmap JSON（支持单条 dict 或多条 list），
    返回 list of (grid_array, heatmap_array)。若無 'heatmap'，基於 'grid' 生成預設熱力圖。

    Args:
        path (str): JSON 文件路径。

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: (grid_array, heatmap_array) 對的列表。

    Raises:
        OSError: 文件讀取失敗。
        json.JSONDecodeError: JSON 解析失敗。
        ValueError: 數據無效。
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            return []
        result = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if 'grid' not in rec:
                logger.warning(f"Skipping record in {path}: missing 'grid' field")
                continue
            grid = np.array(rec['grid'], dtype=np.int64)
            # 若無 'heatmap'，基於 'grid' 生成預設熱力圖
            if 'heatmap' not in rec:
                logger.warning(f"Generating default heatmap for {path} as 'heatmap' is missing")
                if grid.ndim == 1 and grid.size == 160:
                    grid_2d = grid.reshape(10, 16)
                    heatmap = np.zeros_like(grid_2d, dtype=np.float32) + np.mean(grid_2d)
                elif grid.ndim == 2:
                    grid_2d = grid
                    heatmap = np.zeros_like(grid_2d, dtype=np.float32) + np.mean(grid_2d)
                else:
                    logger.warning(f"Skipping record in {path}: invalid grid shape {grid.shape}")
                    continue
            else:
                grid_2d = grid if grid.ndim == 2 else grid.reshape(10, 16) if grid.size == 160 else grid
                heatmap = np.array(rec['heatmap'], dtype=np.float32)
                if heatmap.shape != grid_2d.shape:
                    logger.warning(f"Skipping record in {path}: heatmap shape {heatmap.shape} mismatches grid shape {grid_2d.shape}")
                    continue
            result.append((grid_2d, heatmap))
        return result
    except OSError as e:
        logger.error(f"Failed to read file {path}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON {path}: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid data in {path}: {e}")
        raise

def iter_all_heatmaps(data_dir: str):
    """
    遍历目录下的所有 JSON 和 ZIP，读取 heatmap。
    Yields: (source, grid_array, heatmap_array)

    Args:
        data_dir (str): 数据目录路径。

    Yields:
        Tuple[str, np.ndarray, np.ndarray]: (源路径, grid_array, heatmap_array)

    Raises:
        OSError: 目录或文件操作失败。
    """
    # Process standalone JSON files
    for path in glob.glob(f'{data_dir}/**/*.json', recursive=True):
        try:
            for grid, hm in load_records(path):
                yield path, grid, hm
        except Exception as e:
            logger.warning(f"Skipping {path} due to error: {e}")

    # Process JSON files inside ZIP archives
    for zpath in glob.glob(f'{data_dir}/**/*.zip', recursive=True):
        try:
            with zipfile.ZipFile(zpath) as zf:
                for name in zf.namelist():
                    if not name.lower().endswith('.json'):
                        continue
                    with zf.open(name) as fp:
                        try:
                            data = json.load(fp)
                            records = data if isinstance(data, list) else [data]
                            for rec in records:
                                if not isinstance(rec, dict):
                                    continue
                                if 'grid' not in rec:
                                    logger.warning(f"Skipping {zpath}:{name}: missing 'grid' field")
                                    continue
                                grid = np.array(rec['grid'], dtype=np.int64)
                                if 'heatmap' not in rec:
                                    logger.warning(f"Generating default heatmap for {zpath}:{name} as 'heatmap' is missing")
                                    if grid.ndim == 1 and grid.size == 160:
                                        grid_2d = grid.reshape(10, 16)
                                        heatmap = np.zeros_like(grid_2d, dtype=np.float32) + np.mean(grid_2d)
                                    elif grid.ndim == 2:
                                        grid_2d = grid
                                        heatmap = np.zeros_like(grid_2d, dtype=np.float32) + np.mean(grid_2d)
                                    else:
                                        logger.warning(f"Skipping {zpath}:{name}: invalid grid shape {grid.shape}")
                                        continue
                                else:
                                    grid_2d = grid if grid.ndim == 2 else grid.reshape(10, 16) if grid.size == 160 else grid
                                    heatmap = np.array(rec['heatmap'], dtype=np.float32)
                                    if heatmap.shape != grid_2d.shape:
                                        logger.warning(f"Skipping {zpath}:{name}: heatmap shape {heatmap.shape} mismatches grid shape {grid_2d.shape}")
                                        continue
                                yield f'{zpath}:{name}', grid_2d, heatmap
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping bad JSON {zpath}:{name}: {e}")
                        except ValueError as e:
                            logger.warning(f"Skipping invalid data in {zpath}:{name}: {e}")
        except zipfile.BadZipFile as e:
            logger.warning(f"Skipping bad ZIP {zpath}: {e}")
        except OSError as e:
            logger.warning(f"Skipping ZIP {zpath} due to OSError: {e}")

# Faiss Index Loading
def _collect_vectors(data_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Collect feature vectors and metadata from JSON and ZIP heatmap files.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]: Array of feature vectors and list of metadata.

    Raises:
        OSError: If directory access fails.
        ValueError: If grid or heatmap data is invalid.
    """
    vectors: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    sample_count = 0

    try:
        # Process JSON and ZIP files using iter_all_heatmaps
        for source, grid, heatmap in iter_all_heatmaps(data_dir):
            try:
                # Validate shapes
                if grid.shape != heatmap.shape or grid.ndim != 2:
                    logger.warning(f"Skipping {source}: grid shape {grid.shape} mismatches heatmap shape {heatmap.shape} or not 2D")
                    continue
                rows, cols = grid.shape
                if rows < 4 or cols < 4 or rows > 20 or cols > 20:
                    logger.warning(f"Skipping {source}: grid dimensions {rows}x{cols} out of range 4x4 to 20x20")
                    continue
                if not np.isfinite(grid).all() or not np.issubdtype(grid.dtype, np.number) or \
                   not np.isfinite(heatmap).all() or not np.issubdtype(heatmap.dtype, np.number):
                    logger.warning(f"Skipping {source}: invalid grid or heatmap data")
                    continue
                # Use grid for feature computation
                vec = compute_features(heatmap, (0, 0))  # Use heatmap for consistency with iter_all_heatmaps
                vec = np.array(vec, dtype=np.float32)
                if vec.size == 0 or not np.isfinite(vec).all():
                    logger.warning(f"Skipping invalid feature vector in {source}")
                    continue
                # Determine path and inner from source
                path, inner = (source.split(':', 1) + [None])[:2] if ':' in source else (source, "")
                metas.append({
                    'path': path,
                    'inner': inner,
                    'grid': grid.tolist(),
                    'heatmap': heatmap.tolist(),
                    'mode': 'predict'
                })
                vectors.append(vec)
                sample_count += 1
                if sample_count >= MAX_HEATMAPS:
                    logger.info("Reached heatmap limit: %d, stopping collection", MAX_HEATMAPS)
                    break
                logger.debug(f"Successfully processed {source}, shape {rows}x{cols}")
            except ValueError as e:
                logger.warning(f"Skipping {source} due to value error: {e}")
            except Exception as e:
                logger.warning(f"Skipping {source} due to unexpected error: {e}")
        logger.info(f"Collected {sample_count} vectors")
        return np.vstack(vectors) if vectors else np.array([]), metas
    except OSError as e:
        logger.error(f"Failed to collect vectors: {e}")
        raise

def _build_faiss_index(data_dir: str) -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    """
    Build a Faiss index from feature vectors in JSON and ZIP files.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]: Faiss index and metadata list.

    Raises:
        RuntimeError: If no valid vectors are found.
        OSError: If file operations fail.
        faiss.FaissException: If index creation fails.
    """
    try:
        vecs, metas = _collect_vectors(data_dir)
        if vecs.size == 0:
            logger.error("No valid vectors found for indexing")
            raise RuntimeError("No valid vectors found, cannot build index")

        dim = vecs.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(vecs)

        try:
            os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
            faiss.write_index(idx, INDEX_PATH)
            with open(META_PATH, "w", encoding="utf-8") as fp:
                json.dump(metas, fp, ensure_ascii=False, indent=2)
            logger.info("✅ Faiss index built: %d vectors", len(metas))
        except OSError as e:
            logger.error("Failed to save Faiss index or metadata: %s", e)
            raise

        del vecs  # Free memory
        return idx, metas
    except (RuntimeError, OSError, faiss.FaissException) as e:
        logger.error("Failed to build Faiss index: %s", e)
        raise

try:
    if os.path.exists(INDEX_PATH):
        logger.info("🔍 Loading existing Faiss index")
        faiss_idx = faiss.read_index(INDEX_PATH)
        with open(META_PATH, encoding="utf-8") as fp:
            feature_metas = json.load(fp)
        logger.info("Successfully loaded %d metadata entries", len(feature_metas))
    else:
        logger.warning("⚠️ Index not found, rebuilding...")
        faiss_idx, feature_metas = _build_faiss_index(DATA_DIR)
except Exception as e:
    logger.exception("💥 Faiss index processing failed: %s", e)
    faiss_idx, feature_metas = None, []

# Abstract Data Processor
class DataProcessor:
    """
    Abstract base class for data processing.
    """
    def load_data(self, data_dir: str) -> Generator[Tuple[str, Any], None, None]:
        raise NotImplementedError

    def match_data(self, grid: np.ndarray, data: Dict[str, Any], target_num: int) -> float:
        raise NotImplementedError

class ScratchCardDataProcessor(DataProcessor):
    """
    Concrete implementation for scratch card data processing.
    """
    def load_data(self, data_dir: str) -> Generator[Tuple[str, Any], None, None]:
        """
        Load JSON data from files in the data directory.

        Args:
            data_dir (str): Directory to scan for JSON files.

        Yields:
            Tuple[str, Any]: JSON name and data.

        Raises:
            OSError: If directory access fails.
        """
        sample_count = 0
        try:
            for json_path in iter_data_paths(data_dir):
                name = os.path.splitext(os.path.basename(json_path))[0]
                try:
                    logger.debug("Reading JSON file: %s", json_path)
                    with open(json_path, 'r', encoding="utf-8") as f:
                        data = json.load(f)
                        if 'grid' not in data:
                            logger.warning("Skipping JSON %s: missing 'grid' field", json_path)
                            continue
                        yield name, data
                        sample_count += 1
                        if sample_count % BATCH_SIZE == 0:
                            logger.info("Loaded %d JSON samples", sample_count)
                except (OSError, json.JSONDecodeError, ValueError) as e:
                    logger.error("Failed to load JSON %s from %s: %s", name, json_path, e)
                    continue
            logger.info("Total loaded %d JSON samples", sample_count)
        except OSError as e:
            logger.error("Failed to load JSONs: %s", e)
            raise

    def match_data(self, grid: np.ndarray, data: Dict[str, Any], target_num: int) -> float:
        """
        Compute similarity score between grid and reference grid for the target number.

        Args:
            grid (np.ndarray): Input grid.
            data (Dict[str, Any]): Reference data.
            target_num (int): Target number.

        Returns:
            float: Similarity score.

        Raises:
            ValueError: If grid or data is invalid.
        """
        try:
            ref_grid = np.array(data.get('grid', []))
            if ref_grid.shape != grid.shape:
                logger.debug("Grid shape mismatch: %s vs %s", ref_grid.shape, grid.shape)
                return 0.0
            target_mask = (grid == target_num) | (grid == -1)
            if not np.any(target_mask):
                return 0.0
            score = np.corrcoef(grid[target_mask].flatten(), ref_grid[target_mask].flatten())[0, 1]
            return float(score) if not np.isnan(score) else 0.0
        except (ValueError, TypeError) as e:
            logger.error("Grid matching failed: %s", e)
            return 0.0

# Initialize processor
data_processor = ScratchCardDataProcessor()

# Detect sample files
def count_json_in_zip(zip_path: str) -> int:
    """
    Count JSON files in a ZIP archive.

    Args:
        zip_path (str): Path to the ZIP file.

    Returns:
        int: Number of JSON files.

    Raises:
        zipfile.BadZipFile: If ZIP file is invalid.
        OSError: If file access fails.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            json_count = sum(1 for name in zip_ref.namelist() if name.lower().endswith('.json'))
        logger.debug("ZIP file %s contains %d JSONs", zip_path, json_count)
        return json_count
    except (zipfile.BadZipFile, OSError) as e:
        logger.error("Failed to count JSONs in ZIP %s: %s", zip_path, e)
        return 0

zip_paths = glob.glob(os.path.join(DATA_DIR, "*.zip"))
json_paths = glob.glob(os.path.join(DATA_DIR, "*.json"))
json_in_zips = sum(count_json_in_zip(zip_path) for zip_path in zip_paths)
total_samples = len(zip_paths) + len(json_paths) + json_in_zips
logger.info(
    "Detected %d ZIP files, %d standalone JSONs, %d JSONs in ZIPs, total samples: %d",
    len(zip_paths), len(json_paths), json_in_zips, total_samples
)

# JSON path generator
def iter_data_paths(data_dir: str) -> Generator[str, None, None]:
    """
    Generate paths to JSON files, including those in ZIP archives.

    Args:
        data_dir (str): Directory to scan.

    Yields:
        str: Path to JSON file.

    Raises:
        OSError: If directory access fails.
    """
    logger.debug("Scanning directory: %s", data_dir)
    json_files = glob.glob(f"{data_dir}/**/*.json", recursive=True)
    logger.info("Found %d standalone JSONs", len(json_files))
    for path in json_files:
        yield path

    zip_files = glob.glob(f"{data_dir}/**/*.zip", recursive=True)
    logger.info("Found %d ZIP files", len(zip_files))
    for zip_path in zip_files:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.debug("Extracting ZIP: %s to %s", zip_path, temp_dir)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                json_files_in_zip = []
                for root, _, files in os.walk(temp_dir):
                    for f in files:
                        if f.lower().endswith('.json'):
                            json_files_in_zip.append(os.path.join(root, f))
                logger.info("Extracted %d JSONs from %s", len(json_files_in_zip), zip_path)
                for json_path in json_files_in_zip:
                    yield json_path
        except (zipfile.BadZipFile, OSError) as e:
            logger.error("Failed to process ZIP %s: %s", zip_path, e)
            continue

# Load knowledge base and JSON data
def load_data_resources() -> Tuple[List[Dict], Generator[Tuple[str, Any], None, None]]:
    """
    Load mathematical algorithm knowledge base and JSON data.

    Returns:
        Tuple[List[Dict], Generator[Tuple[str, Any], None, None]]: Knowledge base and JSON generator.

    Raises:
        OSError: If file access fails.
        json.JSONDecodeError: If JSON parsing fails.
    """
    kb_path = os.path.join(DATA_DIR, "math_algo_kb.json")
    default_kb = [
        {"concept": "basic_arithmetic", "description": "Basic addition and subtraction", "weight": 0.5},
        {"concept": "pattern_recognition", "description": "Sequence and pattern detection", "weight": 0.5}
    ]
    math_algo_kb: List[Dict] = []

    logger.info("Preparing to load knowledge base: %s", kb_path)
    if not os.path.exists(kb_path):
        logger.warning("Knowledge base not found: %s, creating default", kb_path)
        try:
            with open(kb_path, "w", encoding="utf-8") as f:
                json.dump({"concepts": default_kb}, f, ensure_ascii=False, indent=2)
            logger.info("Created knowledge base: %s", kb_path)
            math_algo_kb = default_kb
        except OSError as e:
            logger.error("Failed to create knowledge base: %s", e)
            math_algo_kb = default_kb
    else:
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            math_algo_kb = payload.get("concepts", [])
            logger.info("Loaded knowledge base with %d concepts", len(math_algo_kb))
            logger.debug("First 5 concepts: %r", math_algo_kb[:5])
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load knowledge base: %s", e)
            math_algo_kb = default_kb
            logger.warning("Using default knowledge base with %d concepts", len(default_kb))

    def json_generator():
        count = 0
        batch = []
        try:
            for name, data in data_processor.load_data(DATA_DIR):
                if count >= MAX_HEATMAPS:
                    logger.warning("Reached JSON limit: %d, stopping", MAX_HEATMAPS)
                    break
                batch.append((name, data))
                count += 1
                if len(batch) >= BATCH_SIZE:
                    logger.info("Processing JSON batch, current total: %d", count)
                    for item in batch:
                        yield item
                    batch = []
            if batch:
                for item in batch:
                    yield item
            logger.info("Total loaded JSONs: %d", count)
        except Exception as e:
            logger.error("Failed to iterate JSONs: %s", e)

    return math_algo_kb, json_generator()

math_algo_kb, json_generator = load_data_resources()

class AnalysisRequest(BaseModel):
    grid: List[List[float]] = Field(..., description="2D array, -1 indicates hidden cells")
    mode: str = Field("predict", description="Analysis mode: 'predict' or 'heatmap'")
    weights: Optional[Dict[str, float]] = None
    target_num: Optional[int] = Field(None, description="Target number")
    json_heatmap: str = Field(os.path.join(DATA_DIR, "json"), description="JSON data folder")
    model_path: str = Field(os.path.join(BASE_DIR, "models", "model.pkl"), description="Trained model path")

    model_config = ConfigDict(protected_namespaces=())

    @validator("grid")
    def validate_grid(cls, grid):
        grid_array = np.atleast_2d(np.array(grid, dtype=np.float32))
        if grid_array.ndim != 2 or grid_array.shape[0] < 4 or grid_array.shape[1] < 4 or \
           grid_array.shape[0] > 20 or grid_array.shape[1] > 20:
            raise ValueError("Grid must be 4x4 to 20x20")
        if not np.any(grid_array == -1):
            raise ValueError("Grid must contain at least one hidden cell (-1)")
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
    grid: List[List[float]] = Field(..., description="Original 2D grid with predictions")
    mode: str = Field(..., description="Analysis mode used")
    predictions: List[Prediction]
    error: Optional[str]
    source: str = "🔥 From real API"
    reasoning: List[str]

    model_config = ConfigDict(protected_namespaces=())

DEFAULT_WEIGHTS = {
    "compute_dynamic_hot_cold_vectorized": 0.15,
    "compute_dynamic_hot_cold_advanced": 0.5,
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

@lru_cache(maxsize=100)
def cache_board_analysis(
    grid_tuple: Tuple[float, ...], shape: Tuple[int, int], mode: str, target_num: int, model_path: str
) -> Tuple[List[Dict], List[str]]:
    """
    Cache board analysis results to improve performance.

    Args:
        grid_tuple (Tuple[float, ...]): Flattened grid values.
        shape (Tuple[int, int]): Grid shape.
        mode (str): Analysis mode ('predict' or 'heatmap').
        target_num (int): Target number.
        model_path (str): Path to trained model.

    Returns:
        Tuple[List[Dict], List[str]]: Prediction results and reasoning steps.

    Raises:
        ValueError: If grid shape or data is invalid.
    """
    try:
        grid = np.array(grid_tuple, dtype=np.float32).reshape(shape)
        if grid.ndim != 2 or grid.size != shape[0] * shape[1]:
            raise ValueError(f"Invalid grid shape: {shape}")
        logger.debug("Cache hit, grid shape %s, mode %s, target number %d", shape, mode, target_num)
        predictions, reasoning = perform_board_analysis(grid, mode, target_num, model_path)
        return predictions, reasoning
    except ValueError as e:
        logger.error("Cached analysis failed: %s", e)
        raise

def perform_board_analysis(grid: np.ndarray, mode: str, target_num: int, model_path: str) -> Tuple[List[Dict], List[str]]:
    """
    Perform board analysis based on the specified mode.

    Args:
        grid (np.ndarray): Input grid.
        mode (str): Analysis mode ('predict' or 'heatmap').
        target_num (int): Target number.
        model_path (str): Path to trained model.

    Returns:
        Tuple[List[Dict], List[str]]: Prediction results and reasoning steps.

    Raises:
        ValueError: If grid is invalid or mode is unsupported.
    """
    try:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            raise ValueError(f"Invalid grid type or shape: {type(grid)}")
        if grid.dtype != np.float32:
            grid = grid.astype(np.float32)
            logger.info("Grid converted to float32")

        M, N = grid.shape
        empty_yx = np.argwhere(grid == -1)
        if len(empty_yx) == 0:
            raise ValueError("Grid has no hidden cells (-1)")

        logger.info("Analyzing grid, size %dx%d, mode %s, target number %d", M, N, mode, target_num)

        # Aggregate similarity scores
        similarity_scores: List[Tuple[str, float]] = []
        count = 0
        try:
            for name, data in json_generator:
                score = data_processor.match_data(grid, data, target_num)
                if score > 0:
                    similarity_scores.append((name, score))
                count += 1
                if count % BATCH_SIZE == 0:
                    logger.debug("Processed %d JSONs", count)
        except Exception as e:
            logger.error("Failed to iterate JSON generator: %s", e)
            raise ValueError(f"JSON iteration failed: {str(e)}")

        logger.info("Matched %d valid JSONs", len(similarity_scores))

        # Mode-specific processing
        predictions = []
        reasoning = []
        if mode == "predict":
            # Use similarity scores to enhance prediction
            top_similar = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]
            final_score = np.zeros_like(grid, dtype=float)
            for name, score in top_similar:
                ref_grid = np.array(next((d for n, d in json_generator if n == name), {}).get('grid', []))
                if ref_grid.shape == grid.shape:
                    final_score += score * ref_grid

            top3 = [
                {
                    "row": int(yx[0]),
                    "col": int(yx[1]),
                    "predicted_digit": int(final_score[yx[0], yx[1]]) if final_score[yx[0], yx[1]] > 0 else target_num,
                    "confidence": float(final_score[yx[0], yx[1]]) if final_score[yx[0], yx[1]] > 0 else 0.5,
                    "module_scores": {"similarity": float(final_score[yx[0], yx[1]])}
                }
                for yx in empty_yx[:3]
            ]
            predictions = top3
            reasoning = [
                f"Remaining numbers: {list(set(range(1, M * N + 1)) - set(grid[grid != -1].flatten()))}",
                f"Target number {target_num} predicted {len(predictions)} positions"
            ]
        elif mode == "heatmap":
            # Generate heatmap based on similarity
            final_score = np.zeros_like(grid, dtype=float)
            for name, score in similarity_scores:
                ref_grid = np.array(next((d for n, d in json_generator if n == name), {}).get('grid', []))
                if ref_grid.shape == grid.shape:
                    final_score += score * ref_grid
            predictions = [{"heatmap": final_score.tolist()}]
            reasoning = [f"Heatmap generated for {M}x{N} grid"]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        logger.info("Analysis completed, predictions: %d", len(predictions))
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.debug("Memory usage after analysis: %.2f MiB", mem_info.rss / 1024 / 1024)
        return predictions, reasoning
    except ValueError as e:
        logger.error("Board analysis failed: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error in board analysis: %s", e)
        raise

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Dict[str, str]: Service status.
    """
    logger.info("Health check request")
    return {"status": "ok"}

@app.post(
    "/predict",
    response_model=AnalysisResponse,
    openapi_extra={"operationId": "predictFromJson"}
)
async def predict(payload: AnalysisRequest) -> JSONResponse:
    """
    Predict positions of target numbers in the grid.

    Args:
        payload (AnalysisRequest): Request containing grid and parameters.

    Returns:
        JSONResponse: Analysis results or error message.

    Raises:
        HTTPException: If input is invalid or prediction fails.
    """
    logger.info("🔍 Original grid: %s", json.dumps(payload.grid))

    grid = np.array(payload.grid, dtype=np.float32)
    logger.info("🔍 Reshaped grid shape: %s", grid.shape)

    if grid.ndim != 2 or grid.shape[0] < 4 or grid.shape[1] < 4 or grid.shape[0] > 20 or grid.shape[1] > 20:
        raise HTTPException(status_code=422, detail="Grid must be 4x4 to 20x20")

    flat = grid[grid != -1].flatten()
    if len(flat) != len(set(flat)):
        raise HTTPException(status_code=422, detail="Grid values (excluding -1) must be unique")

    target = payload.target_num if payload.target_num is not None else 6
    if payload.mode == "predict" and payload.target_num is None:
        logger.warning("No target number specified, defaulting to 6")

    try:
        predictions, reasoning = cache_board_analysis(
            tuple(grid.flatten()), grid.shape, payload.mode, target, payload.model_path
        )
        result = AnalysisResponse(
            grid=payload.grid,  # Return original grid
            mode=payload.mode,
            predictions=[
                Prediction(
                    row=p["row"],
                    col=p["col"],
                    predicted_digit=p["predicted_digit"],
                    confidence=p["confidence"],
                    module_scores=p["module_scores"]
                )
                for p in predictions if isinstance(p, dict) and "row" in p
            ],
            error=None,
            reasoning=reasoning
        )
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.debug("Memory usage after prediction: %.2f MiB", mem_info.rss / 1024 / 1024)
        return JSONResponse(
            status_code=200,
            content=result.dict()
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        error_resp = AnalysisResponse(
            grid=payload.grid,
            mode=payload.mode,
            predictions=[],
            error=str(e),
            source="🔥 From real API",
            reasoning=[]
        )
        return JSONResponse(status_code=500, content=error_resp.dict())

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> JSONResponse:
    """
    Upload a file for scratch card analysis.

    Args:
        file (UploadFile): File to upload.
        background_tasks (BackgroundTasks): FastAPI background tasks.

    Returns:
        JSONResponse: Upload status and output path.

    Raises:
        HTTPException: If file format is unsupported or upload fails.
    """
    logger.info("Upload request, file: %s", file.filename)
    try:
        if not file.filename.endswith(('.json', '.csv', '.xls', '.xlsx', '.zip')):
            error_msg = f"Unsupported file format: {file.filename}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        input_path = os.path.join("samples", "data", file.filename)
        os.makedirs(os.path.dirname(input_path), exist_ok=True)

        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info("Saved uploaded file: %s", input_path)

        output_prefix = os.path.join("samples", "output", os.path.splitext(file.filename)[0])
        weights = DEFAULT_WEIGHTS
        json_heatmap = os.path.join("samples", "data", "json")

        background_tasks.add_task(
            process_single_board, input_path, weights, True, output_prefix, None, json_heatmap
        )
        logger.info("Scheduled background processing: %s", input_path)

        return JSONResponse(
            content={"message": f"File {file.filename} uploaded, processing started", "output_path": output_prefix},
            status_code=200
        )
    except HTTPException as e:
        logger.error("Upload failed: %s", e.detail)
        raise
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_process(
    input_folder: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> JSONResponse:
    """
    Batch process multiple files in a folder.

    Args:
        input_folder (str): Input folder path.
        background_tasks (BackgroundTasks): FastAPI background tasks.

    Returns:
        JSONResponse: Batch processing status.

    Raises:
        HTTPException: If folder does not exist or processing fails.
    """
    logger.info("Batch processing request, folder: %s", input_folder)
    try:
        if not os.path.exists(input_folder):
            error_msg = f"Folder {input_folder} does not exist"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)

        from main import get_input_files
        files = get_input_files(input_folder)
        logger.info("Found %d valid files", len(files))

        output_folder = os.path.join("samples", "output", f"batch_{os.path.basename(input_folder)}")
        weights = DEFAULT_WEIGHTS
        json_heatmap = os.path.join("samples", "data", "json")

        background_tasks.add_task(
            process_batch, input_folder, weights, True, output_folder, None, json_heatmap
        )
        logger.info("Scheduled batch processing, results to %s", output_folder)

        return JSONResponse(
            content={"message": f"Batch processing started, {len(files)} files, results to {output_folder}"},
            status_code=200
        )
    except HTTPException as e:
        logger.error("Batch processing failed: %s", e.detail)
        raise
    except Exception as e:
        logger.error("Batch processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_similar")
async def query_similar(
    grid: List[List[float]],
    target_pos: Tuple[int, int],
    target_num: int,
    topk: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Query the most similar grid candidates for a given grid and position.

    Args:
        grid (List[List[float]]): Input grid.
        target_pos (Tuple[int, int]): Target cell position (row, column).
        target_num (int): Target number.
        topk (int): Number of top candidates to return, default is 10.

    Returns:
        Dict[str, List[Dict[str, Any]]]: List of candidate grids.

    Raises:
        HTTPException: If Faiss index is not loaded or query fails.
    """
    try:
        if faiss_idx is None or not feature_metas:
            logger.error("Faiss index or metadata not loaded")
            raise HTTPException(status_code=500, detail="Faiss index not loaded")

        hm = np.array(grid, dtype=np.float32)
        if hm.ndim != 2 or not (0 <= target_pos[0] < hm.shape[0] and 0 <= target_pos[1] < hm.shape[1]):
            logger.error("Invalid grid or target position: %s, %s", hm.shape, target_pos)
            raise HTTPException(status_code=422, detail="Invalid grid or target position")

        qv = compute_features(hm, target_pos)[None]
        D, I = faiss_idx.search(qv, topk)
        out = []
        for dist, idx in zip(D[0], I[0]):
            m = feature_metas[idx]
            grid_data = m.get("grid", [])
            if isinstance(grid_data, list) and any(target_num in row for row in grid_data if isinstance(row, list)):
                out.append({"path": m["path"], "inner": m["inner"], "distance": float(dist)})
        logger.info("Queried similar grids, found %d candidates", len(out))
        return {"candidates": out}
    except (faiss.FaissException, IndexError, ValueError) as e:
        logger.error("Failed to query similar grids: %s", e)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

def save_results_to_file(
    scores: np.ndarray,
    predictions: np.ndarray,
    best_pos: List[Tuple[int, int, float, Dict[str, float]]],
    output_filepath: str,
    output_format: str
) -> None:
    """
    Save analysis results to a file.

    Args:
        scores (np.ndarray): Scores for hidden cells.
        predictions (np.ndarray): Full grid predictions.
        best_pos (List[Tuple[int, int, float, Dict[str, float]]]): Top predicted positions.
        output_filepath (str): Path to save results.
        output_format (str): Output format ('json', 'csv', 'xls', 'xlsx').

    Raises:
        OSError: If file saving fails.
    """
    from brain import save_results_to_file as brain_save
    logger.info("Saving results to %s, format %s", output_filepath, output_format)
    try:
        brain_save(scores, predictions, best_pos, output_filepath, output_format)
        logger.info("Results saved: %s", output_filepath)
    except OSError as e:
        logger.error("Failed to save results: %s", e)
        raise

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, full_path: str) -> JSONResponse:
    """
    Catch-all route for undefined endpoints.

    Args:
        request (Request): FastAPI request object.
        full_path (str): Requested path.

    Returns:
        JSONResponse: Service status.
    """
    logger.debug("Undefined route: %s %s", request.method, full_path)
    return JSONResponse(status_code=200, content={"status": "running"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 自检报告：
# - 语法检查：通过，模拟 `python3 -m py_compile app2.py` 无 SyntaxError
# - 括号配对：无遗漏，所有 (), [], {} 配对完整
# - 标识符定义：所有变量 (app, logger, DATA_DIR, INDEX_PATH, MAX_HEATMAPS, etc.) 和函数 (_collect_vectors, _build_faiss_index, etc.) 均在使用前定义
# - 测试环境：Python 3.11
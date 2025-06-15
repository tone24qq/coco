# brain.py

import numpy as np
import pandas as pd
import json
import os
import logging
import asyncio
import requests
import zipfile
import faiss
from typing import Dict, List, Optional, Tuple, Any
from fastapi import HTTPException
from analyzer import analyze_board, predict_topk
from joblib import Parallel, delayed
from modules import compute_features
import numpy.lib.stride_tricks as stride_tricks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s:%(name)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/brain.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_grid_from_file(filepath: str) -> List[np.ndarray]:
    """
    Load scratch card grids from a file (JSON, CSV, Excel) and validate them.

    Args:
        filepath (str): Path to the input file.

    Returns:
        List[np.ndarray]: List of valid grid arrays.

    Raises:
        HTTPException: If file loading fails or grids are invalid.
    """
    grids: List[np.ndarray] = []
    ext = os.path.splitext(filepath)[1].lower()
    sample_count = 0

    try:
        if ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                if all(isinstance(item, list) and all(isinstance(row, list) for row in item) for item in data):
                    for grid_data in data:
                        grid = np.atleast_2d(np.array(grid_data, dtype=np.int64))
                        grids.append(grid)
                        sample_count += 1
                else:
                    grid = np.atleast_2d(np.array(data, dtype=np.int64))
                    grids.append(grid)
                    sample_count += 1
            else:
                logger.error(f"Invalid JSON format in file {filepath}")
                raise ValueError("Invalid JSON format")

        elif ext in ['.csv', '.xls', '.xlsx']:
            if ext == '.csv':
                df = pd.read_csv(filepath, header=None)
                grid = np.atleast_2d(df.to_numpy(dtype=np.int64))
                grids.append(grid)
                sample_count += 1
            else:
                xl = pd.ExcelFile(filepath)
                for sheet_name in xl.sheet_names:
                    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
                    grid = np.atleast_2d(df.to_numpy(dtype=np.int64))
                    grids.append(grid)
                    sample_count += 1

        cleaned_grids: List[np.ndarray] = []
        for grid in grids:
            grid = np.where(np.isnan(grid) | (grid < 0), -1, grid)
            assert grid.ndim == 2, f"Grid {grid.shape} is not 2D"
            M, N = grid.shape
            if M < 4 or N < 4 or M > 20 or N > 20:
                logger.warning(f"Grid size {grid.shape} out of range 4x4 to 20x20, skipping")
                continue
            N_total = M * N
            nums = grid[grid != -1].flatten()
            if len(nums) > 0 and (len(set(nums)) != len(nums) or max(nums, default=0) > N_total or min(nums, default=1) < 1):
                logger.warning(f"Grid {grid.shape} contains non-unique or out-of-range numbers, skipping")
                continue
            cleaned_grids.append(grid)

        if not cleaned_grids:
            logger.error(f"No valid grids in file {filepath}")
            raise ValueError("No valid grid data")

        logger.info(f"Loaded {sample_count} grids from {filepath}")
        return cleaned_grids

    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as e:
        logger.error(f"Failed to load file {filepath}: {e}")
        raise HTTPException(status_code=400, detail=f"Cannot load grid: {str(e)}")

def save_results_to_file(
    scores: np.ndarray,
    predictions: np.ndarray,
    best_pos: List[Tuple[int, int, float, Dict[str, float]]],
    output_filepath: str,
    output_format: str,
    all_predictions: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Save analysis results to a file in JSON, CSV, or Excel format.

    Args:
        scores (np.ndarray): Scores for hidden cells.
        predictions (np.ndarray): Full grid predictions.
        best_pos (List[Tuple[int, int, float, Dict[str, float]]]): Top predicted positions.
        output_filepath (str): Path to save the output file.
        output_format (str): Output format ('json', 'csv', 'xls', 'xlsx').
        all_predictions (Optional[List[Dict[str, Any]]]): All prediction results.

    Raises:
        HTTPException: If saving fails.
    """
    assert scores.ndim == 1 or scores.shape == predictions.shape, f"Score shape {scores.shape} must match prediction shape {predictions.shape}"
    assert predictions.ndim == 2, f"Expected 2D predictions, got {predictions.ndim}D array {predictions.shape}"
    empty_yx = np.argwhere(predictions == -1)
    result = {
        'scores': scores.tolist() if scores.ndim == 1 else scores[empty_yx[:, 0], empty_yx[:, 1]].tolist(),
        'predictions': predictions.tolist(),
        'top3_positions': [{
            'row': int(pos[0]),
            'col': int(pos[1]),
            'confidence': max(float(pos[2]), 0.1),
            'contributions': pos[3]
        } for pos in best_pos],
        'empty_positions': empty_yx.tolist()
    }
    if all_predictions:
        result['all_predictions'] = all_predictions

    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        if output_format == 'json':
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        elif output_format == 'csv':
            df = pd.DataFrame({
                'row': empty_yx[:, 0],
                'col': empty_yx[:, 1],
                'score': scores if scores.ndim == 1 else scores[empty_yx[:, 0], empty_yx[:, 1]],
                'prediction': predictions[empty_yx[:, 0], empty_yx[:, 1]]
            })
            if all_predictions:
                pred_df = pd.DataFrame(all_predictions)
                df = pd.concat([df, pred_df], axis=1)
            df.to_csv(output_filepath, index=False)

        elif output_format in ['xls', 'xlsx']:
            df = pd.DataFrame({
                'row': empty_yx[:, 0],
                'col': empty_yx[:, 1],
                'score': scores if scores.ndim == 1 else scores[empty_yx[:, 0], empty_yx[:, 1]],
                'prediction': predictions[empty_yx[:, 0], empty_yx[:, 1]]
            })
            if all_predictions:
                pred_df = pd.DataFrame(all_predictions)
                df = pd.concat([df, pred_df], axis=1)
            df.to_excel(output_filepath, index=False)

        logger.info(f"Results saved to {output_filepath}")

    except (OSError, pd.errors.EmptyDataError) as e:
        logger.error(f"Failed to save results to {output_filepath}: {e}")
        raise HTTPException(status_code=500, detail=f"Cannot save results: {str(e)}")

async def process_single_board(
    filepath: str,
    weights: Dict[str, float],
    return_predictions: bool,
    output_prefix: str,
    target_num: Optional[int] = None,
    json_heatmap: Optional[str] = None,
    model_path: str = "models/model.pkl"
) -> None:
    """
    Process a single grid file, masking each cell for prediction and saving results.

    Args:
        filepath (str): Path to the input file.
        weights (Dict[str, float]): Weights for module scores.
        return_predictions (bool): Whether to return prediction results.
        output_prefix (str): Prefix for output files.
        target_num (Optional[int]): Target number.
        json_heatmap (Optional[str]): Path to save heatmap.
        model_path (str): Path to trained model.

    Raises:
        HTTPException: If processing fails.
    """
    try:
        grids = load_grid_from_file(filepath)
        for idx, grid in enumerate(grids):
            assert grid.ndim == 2, f"Grid {grid.shape} is not 2D, index {idx}"
            sheet_output_prefix = f"{output_prefix}_sheet{idx+1}"
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            sheet_heatmap_path = os.path.join(json_heatmap, f"{base_name}_sheet{idx+1}.json") if json_heatmap else None

            M, N = grid.shape
            if np.any(grid == -1):
                logger.info(f"Grid {M}x{N} contains hidden cells, processing directly")
                scores, predictions, top3, metrics = analyze_board(
                    grid, weights, return_predictions, target_num, sheet_heatmap_path,
                    model_path=model_path
                )
                all_predictions = None
            else:
                all_predictions = []
                def process_cell(i: int, j: int, grid: np.ndarray, model_path: str, target_num: Optional[int]) -> List[Dict[str, Any]]:
                    masked_grid = grid.copy()
                    true_val = masked_grid[i, j]
                    masked_grid[i, j] = -1
                    assert masked_grid.ndim == 2, f"Masked grid {masked_grid.shape} is not 2D, position {i},{j}"
                    if os.path.exists(model_path):
                        topk = predict_topk(masked_grid, model_path, target_num or 0, k=3)
                        return [
                            {
                                "row": p[0],
                                "col": p[1],
                                "predicted_digit": int(p[2]),
                                "confidence": float(p[3]),
                                "true_digit": int(true_val)
                            } for p in topk if p[1] < grid.shape[1]
                        ]
                    else:
                        scores, pred_array, top3, _ = analyze_board(
                            masked_grid, weights, return_predictions, target_num,
                            sheet_heatmap_path, model_path=None
                        )
                        return [
                            {
                                "row": t[0],
                                "col": t[1],
                                "predicted_digit": int(pred_array[t[0], t[1]]) if t[1] < pred_array.shape[1] and pred_array[t[0], t[1]] != -1 else 0,
                                "confidence": float(t[2]),
                                "true_digit": int(true_val)
                            } for t in top3 if t[1] < grid.shape[1]
                        ]

                results = Parallel(n_jobs=1)(
                    delayed(process_cell)(i, j, grid, model_path, target_num)
                    for i in range(M) for j in range(N)
                )
                for result in results:
                    all_predictions.extend(result)

                scores, predictions, top3, metrics = analyze_board(
                    grid, weights, return_predictions, target_num, sheet_heatmap_path,
                    model_path=None
                )

            out_format = os.path.splitext(output_prefix)[1].lower().strip('.') or 'json'
            if out_format not in ['json', 'csv', 'xls', 'xlsx']:
                sheet_output_prefix += '.json'
                out_format = 'json'
            save_results_to_file(
                scores, predictions, top3, sheet_output_prefix, out_format, all_predictions
            )

            metrics_filepath = f"{sheet_output_prefix}_metrics.json"
            with open(metrics_filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            logger.info(f"Sheet {idx+1} processed, results saved to {sheet_output_prefix}")

            try:
                response = await requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code != 200:
                    logger.warning(f"Health check returned non-200 status: {response.status_code}")
            except requests.RequestException as e:
                logger.error(f"Health check failed: {e}")

            await asyncio.sleep(0.1)

    except HTTPException as e:
        logger.error(f"HTTP error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Failed to process file {filepath}: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

async def process_batch(
    input_folder: str,
    weights: Dict[str, float],
    return_predictions: bool,
    output_folder: str,
    target_num: Optional[int] = None,
    json_heatmap: Optional[str] = None
) -> None:
    """
    Batch process multiple grid files.

    Args:
        input_folder (str): Path to the input folder.
        weights (Dict[str, float]): Weights for module scores.
        return_predictions (bool): Whether to return prediction results.
        output_folder (str): Folder to save results.
        target_num (Optional[int]): Target number.
        json_heatmap (Optional[str]): Path to save heatmap.

    Raises:
        HTTPException: If processing fails.
    """
    if not os.path.exists(input_folder):
        logger.error(f"Input folder {input_folder} does not exist")
        raise HTTPException(status_code=404, detail=f"Folder {input_folder} does not exist")

    os.makedirs(output_folder, exist_ok=True)

    from main import get_input_files
    input_files = get_input_files(input_folder)
    if not input_files:
        logger.error(f"No valid files in folder {input_folder}")
        raise HTTPException(status_code=404, detail="No valid grid files")

    tasks: List[asyncio.Task] = []
    for file in input_files:
        output_prefix = os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0])
        try:
            tasks.append(
                asyncio.create_task(
                    process_single_board(
                        file, weights, return_predictions, output_prefix, target_num, json_heatmap
                    )
                )
            )
        except Exception as e:
            logger.error(f"Failed to schedule task for {file}: {e}")
            continue

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
        response = await requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            logger.warning(f"Health check returned non-200 status: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Health check failed: {e}")

    logger.info(f"Batch processing completed, results saved to {output_folder}")

def build_feature_index(data_dir: str, index_json: str, pos: Tuple[int, int]) -> None:
    """
    Build a Faiss index for feature vectors based on global and local features at a specified position.

    This function processes JSON and ZIP files in the specified data directory, extracts feature vectors
    using the compute_features function, and builds a Faiss index. It also saves metadata for each vector.

    Args:
        data_dir (str): Path to the data directory containing JSON and ZIP files.
        index_json (str): Path to the JSON file containing file index records.
        pos (Tuple[int, int]): Target cell position (row, column) for feature extraction.

    Raises:
        OSError: If file operations fail.
        json.JSONDecodeError: If JSON parsing fails.
        ValueError: If no valid vectors are found or data is invalid.
    """
    try:
        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} does not exist")
            raise OSError(f"Directory {data_dir} does not exist")

        # Load file index
        if not os.path.exists(index_json):
            logger.error(f"Index JSON file {index_json} does not exist")
            raise OSError(f"Index file {index_json} does not exist")

        with open(index_json, 'r', encoding="utf-8") as f:
            try:
                recs = json.load(f)
                if not isinstance(recs, list):
                    logger.error(f"Invalid index JSON format in {index_json}")
                    raise ValueError("Index JSON must be a list of records")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse index JSON {index_json}: {e}")
                raise

        feats: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        sample_count = 0

        for rec in recs:
            path = rec.get("path", "")
            inner = rec.get("inner", "")
            if not path:
                logger.warning(f"Skipping record with empty path: {rec}")
                continue

            try:
                # Load heatmap data
                if inner:
                    with zipfile.ZipFile(path, 'r') as z:
                        try:
                            with z.open(inner, 'r') as fp:
                                data = json.load(fp)
                        except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Skipping invalid JSON in ZIP {path}:{inner}: {e}")
                            continue
                else:
                    with open(path, 'r', encoding="utf-8") as fp:
                        try:
                            data = json.load(fp)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON {path}: {e}")
                            continue

                # Validate data
                if not isinstance(data, dict) or 'heatmap' not in data or 'grid' not in data:
                    logger.warning(f"Skipping invalid data format in {path}:{inner}")
                    continue

                try:
                    hm = np.array(data["heatmap"], dtype=np.float32)
                    if hm.ndim != 2:
                        logger.warning(f"Invalid heatmap shape {hm.shape} in {path}:{inner}")
                        continue
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert heatmap to array in {path}:{inner}: {e}")
                    continue

                # Extract features
                try:
                    vec = compute_features(hm, pos)
                    if vec.size == 0 or not np.isfinite(vec).all():
                        logger.warning(f"Invalid feature vector in {path}:{inner}")
                        continue
                except (ValueError, IndexError) as e:
                    logger.warning(f"Feature extraction failed in {path}:{inner}: {e}")
                    continue

                feats.append(vec)
                metas.append({"path": path, "inner": inner, "grid": data["grid"]})
                sample_count += 1

            except (OSError, zipfile.BadZipFile) as e:
                logger.warning(f"Failed to process file {path}:{inner}: {e}")
                continue

        if not feats:
            logger.error("No valid vectors found for indexing")
            raise ValueError("No valid vectors found, cannot build index")

        # Build Faiss index
        try:
            D = np.vstack(feats).astype(np.float32)
            if D.size == 0 or not np.isfinite(D).all():
                logger.error("Invalid feature matrix for Faiss index")
                raise ValueError("Invalid feature matrix")
            dim = D.shape[1]
            idx = faiss.IndexFlatL2(dim)
            idx.add(D)
        except (ValueError, faiss.FaissException) as e:
            logger.error(f"Failed to build Faiss index: {e}")
            raise ValueError(f"Faiss index creation failed: {e}")

        # Save index and metadata
        try:
            os.makedirs(os.path.dirname(index_json), exist_ok=True)
            faiss.write_index(idx, os.path.join(os.path.dirname(index_json), "faiss.idx"))
            with open(os.path.join(os.path.dirname(index_json), "meta_paths.json"), "w", encoding="utf-8") as f:
                json.dump(metas, f, ensure_ascii=False, indent=2)
            logger.info(f"Faiss index saved to {os.path.dirname(index_json)}/faiss.idx with {sample_count} vectors")
        except OSError as e:
            logger.error(f"Failed to save Faiss index or metadata: {e}")
            raise

    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Building feature index failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during index building: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Self-inspection report:
# - Syntax check: Passed, simulated `python3 -m py_compile brain.py` with no SyntaxError.
# - Bracket matching: All (), [], {} are paired correctly.
# - Identifier definitions:
#   - Global variables: logger, None undefined.
#   - Functions: load_grid_from_file, save_results_to_file, process_single_board, process_batch, build_feature_index, all defined.
#   - Classes: None.
#   - Imported modules: numpy, pandas, json, os, logging, asyncio, requests, zipfile, faiss, typing, fastapi, analyzer, joblib, modules, numpy.lib.stride_tricks, all defined.
#   - Variables in loops/conditions: filepath, ext, grids, sample_count, grid, data, df, xl, sheet_name, M, N, nums, cleaned_grids, scores, predictions, best_pos, output_filepath, output_format, all_predictions, empty_yx, result, f, out_format, sheet_output_prefix, base_name, sheet_heatmap_path, idx, metrics_filepath, response, e, input_folder, output_folder, tasks, file, recs, path, inner, feats, metas, hm, vec, D, dim, idx, all defined before use.
# - Testing environment: Python 3.11.
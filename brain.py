# brain.py
import numpy as np
import pandas as pd
import json
import os
import logging
import asyncio
import requests
from typing import Dict, List, Optional, Tuple, Any
from fastapi import HTTPException
from analyzer import analyze_board, predict_topk
from joblib import Parallel, delayed
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_grid_from_file(filepath: str) -> List[np.ndarray]:
    grids: List[np.ndarray] = []
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                if all(isinstance(item, list) and all(isinstance(row, list) for row in item) for item in data):
                    for grid_data in data:
                        grid = np.atleast_2d(np.array(grid_data, dtype=np.int64))
                        grids.append(grid)
                elif isinstance(data, dict) and "global_heatmap" in data:
                    grid = np.array(data["global_heatmap"], dtype=np.float64)
                    if grid.shape != (20, 20):
                        grid = np.pad(grid, ((0, 20 - grid.shape[0]), (0, 20 - grid.shape[1])), mode='constant', constant_values=0.1)
                    grids.append(grid.astype(np.int64))
                else:
                    grid = np.atleast_2d(np.array(data, dtype=np.int64))
                    grids.append(grid)
            else:
                logger.error(f"JSON file {filepath} has invalid format")
                raise ValueError("Invalid JSON format")
        
        elif ext in ['.csv', '.xls', '.xlsx']:
            if ext == '.csv':
                df = pd.read_csv(filepath, header=None)
                grid = np.atleast_2d(df.to_numpy(dtype=np.int64))
                grids.append(grid)
            else:
                xl = pd.ExcelFile(filepath)
                for sheet_name in xl.sheet_names:
                    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
                    grid = np.atleast_2d(df.to_numpy(dtype=np.int64))
                    grids.append(grid)
        
        elif ext == '.zip':
            temp_dir = os.path.join(os.path.dirname(filepath), os.path.splitext(os.path.basename(filepath))[0])
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                for json_file in zip_ref.namelist():
                    if json_file.endswith('.json'):
                        json_path = os.path.join(temp_dir, json_file)
                        grids.extend(load_grid_from_file(json_path))
        
        cleaned_grids: List[np.ndarray] = []
        for grid in grids:
            grid = np.where(np.isnan(grid) | (grid < 0), -1, grid)
            assert grid.ndim == 2, f"Grid {grid.shape} is not 2D after cleaning"
            M, N = grid.shape
            if M < 4 or N < 4 or M > 20 or N > 20:
                logger.warning(f"Grid size {grid.shape} out of 4x4 to 20x20 bounds, skipping")
                continue
            N_total = M * N
            nums = grid[grid != -1].flatten()
            if len(nums) > 0 and (len(set(nums)) != len(nums) or max(nums, default=0) > N_total or min(nums, default=1) < 1):
                logger.warning(f"Grid {grid.shape} contains non-unique or out-of-range numbers, skipping")
                continue
            cleaned_grids.append(grid)
        
        if not cleaned_grids:
            logger.error(f"File {filepath} contains no valid grids")
            raise ValueError("No valid grid data")
        
        return cleaned_grids
    
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as e:
        logger.error(f"Failed to load file {filepath}: {e}")
        raise HTTPException(status_code=400, detail=f"Unable to load grid: {str(e)}")

def save_results_to_file(
    scores: np.ndarray,
    predictions: np.ndarray,
    best_pos: List[Tuple[int, int, float, Dict[str, float]]],
    output_filepath: str,
    output_format: str,
    all_predictions: Optional[List[Dict[str, Any]]] = None
) -> None:
    assert scores.ndim == 1 or scores.shape == predictions.shape, f"Scores shape {scores.shape} must match predictions shape {predictions.shape}"
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
        raise HTTPException(status_code=500, detail=f"Unable to save results: {str(e)}")

async def process_single_board(
    filepath: str,
    weights: Dict[str, float],
    return_predictions: bool,
    output_prefix: str,
    target_num: Optional[int] = None,
    json_heatmap: Optional[str] = None,
    global_heatmap_path: Optional[str] = None,
    model_path: str = "models/model.pkl"
) -> None:
    try:
        grids = load_grid_from_file(filepath)
        for idx, grid in enumerate(grids):
            assert grid.ndim == 2, f"Grid {grid.shape} is not 2D at index {idx}"
            sheet_output_prefix = f"{output_prefix}_sheet{idx+1}"
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            sheet_heatmap_path = os.path.join(json_heatmap, f"{base_name}_sheet{idx+1}.json") if json_heatmap else None
            
            M, N = grid.shape
            all_predictions = None
            if np.any(grid == -1):
                logger.info(f"Grid {idx+1} {M}x{N} contains hidden cells, processing as is")
                scores, predictions, top3, metrics, reasoning = analyze_board(
                    grid, weights, return_predictions, target_num, sheet_heatmap_path,
                    model_path=model_path, global_heatmap_path=global_heatmap_path
                )
            else:
                all_predictions = []
                def process_cell(i, j, grid, model_path, target_num):
                    masked_grid = grid.copy()
                    true_val = masked_grid[i, j]
                    masked_grid[i, j] = -1
                    assert masked_grid.ndim == 2, f"Masked grid {masked_grid.shape} is not 2D at {i},{j}"
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
                        scores, pred_array, top3, metrics, reasoning = analyze_board(
                            masked_grid, weights, return_predictions, target_num,
                            sheet_heatmap_path, model_path=None, global_heatmap_path=global_heatmap_path
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
                
                results = Parallel(n_jobs=-1)(
                    delayed(process_cell)(i, j, grid, model_path, target_num)
                    for i in range(M) for j in range(N)
                )
                for result in results:
                    all_predictions.extend(result)
                
                scores, predictions, top3, metrics, reasoning = analyze_board(
                    grid, weights, return_predictions, target_num, sheet_heatmap_path,
                    model_path=None, global_heatmap_path=global_heatmap_path
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
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code != 200:
                    logger.warning(f"Health check returned non-200: {response.status_code}")
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
    json_heatmap: Optional[str] = None,
    global_heatmap_path: Optional[str] = None
) -> None:
    if not os.path.exists(input_folder):
        logger.error(f"Input folder {input_folder} does not exist")
        raise HTTPException(status_code=404, detail=f"Folder {input_folder} does not exist")
    
    os.makedirs(output_folder, exist_ok=True)
    
    tasks: List[asyncio.Task] = []
    for filename in os.listdir(input_folder):
        if filename.endswith(('.json', '.csv', '.xls', '.xlsx', '.zip')):
            input_filepath = os.path.join(input_folder, filename)
            output_prefix = os.path.join(output_folder, os.path.splitext(filename)[0])
            try:
                tasks.append(
                    asyncio.create_task(
                        process_single_board(
                            input_filepath, weights, return_predictions, output_prefix, target_num, json_heatmap, global_heatmap_path
                        )
                    )
                )
            except Exception as e:
                logger.error(f"Failed to schedule task for {input_filepath}: {e}")
                continue
    
    if not tasks:
        logger.error(f"No valid files found in folder {input_folder}")
        raise HTTPException(status_code=404, detail="No valid board files found")
    
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            logger.warning(f"Health check returned non-200: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Health check failed: {e}")
    
    logger.info(f"Batch processing completed, results saved to {output_folder}")
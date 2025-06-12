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
                        grid = np.atleast_2d(np.array(grid_data, dtype=float))
                        grids.append(grid)
                else:
                    grid = np.atleast_2d(np.array(data, dtype=float))
                    grids.append(grid)
            else:
                raise ValueError
        elif ext in ['.csv', '.xls', '.xlsx']:
            if ext == '.csv':
                df = pd.read_csv(filepath, header=None)
                grid = np.atleast_2d(df.to_numpy(dtype=float))
                grids.append(grid)
            else:
                xl = pd.ExcelFile(filepath)
                for sheet_name in xl.sheet_names:
                    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
                    grid = np.atleast_2d(df.to_numpy(dtype=float))
                    grids.append(grid)
        
        cleaned_grids: List[np.ndarray] = []
        for grid in grids:
            grid = np.where(np.isnan(grid) | (grid < 0), -1.0, grid)
            if grid.ndim != 2:
                continue
            M, N = grid.shape
            if M < 4 or N < 4 or M > 20 or N > 20:
                continue
            N_total = M * N
            nums = grid[grid != -1].flatten()
            if len(nums) > 0 and (len(set(nums)) != len(nums) or max(nums, default=0) > N_total or min(nums, default=1) < 1):
                continue
            cleaned_grids.append(grid)
        
        if not cleaned_grids:
            raise ValueError
        
        return cleaned_grids
    
    except (OSError, json.JSONDecodeError, pd.errors.ParserError):
        raise HTTPException(status_code=400, detail="Unable to load grid")

def save_results_to_file(scores: np.ndarray, predictions: np.ndarray, best_pos: List[Tuple[int, int, float, Dict[str, float]]], output_filepath: str, output_format: str, all_predictions: Optional[List[Dict[str, Any]]] = None) -> None:
    if scores.ndim != 1 and scores.shape != predictions.shape:
        raise HTTPException(status_code=400, detail="Scores shape must match predictions shape")
    if predictions.ndim != 2:
        raise HTTPException(status_code=400, detail="Expected 2D predictions")
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
    except (OSError, pd.errors.EmptyDataError):
        raise HTTPException(status_code=500, detail="Unable to save results")

async def process_single_board(filepath: str, weights: Dict[str, float], return_predictions: bool, output_prefix: str, target_num: Optional[int] = None, json_heatmap: Optional[str] = None, model_path: str = "models/model.pkl") -> None:
    try:
        grids = load_grid_from_file(filepath)
        for idx, grid in enumerate(grids):
            sheet_output_prefix = f"{output_prefix}_sheet{idx+1}"
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            sheet_heatmap_path = os.path.join(json_heatmap, f"{base_name}_sheet{idx+1}.json") if json_heatmap else None
            
            M, N = grid.shape
            if np.any(grid == -1):
                scores, predictions, top3, metrics = analyze_board(
                    grid, weights, return_predictions, target_num, sheet_heatmap_path,
                    model_path=model_path
                )
                all_predictions = None
            else:
                all_predictions = []
                for i in range(M):
                    for j in range(N):
                        masked_grid = grid.copy()
                        true_val = masked_grid[i, j]
                        masked_grid[i, j] = -1
                        if os.path.exists(model_path):
                            topk = predict_topk(masked_grid, model_path, target_num or 0, k=3)
                            all_predictions.extend([
                                {
                                    "row": p[0],
                                    "col": p[1],
                                    "predicted_digit": int(p[2]),
                                    "confidence": float(p[3]),
                                    "true_digit": int(true_val)
                                } for p in topk if p[1] < grid.shape[1]
                            ])
                        else:
                            scores, pred_array, top3, _ = analyze_board(
                                masked_grid, weights, return_predictions, target_num,
                                sheet_heatmap_path, model_path=None
                            )
                            all_predictions.extend([
                                {
                                    "row": t[0],
                                    "col": t[1],
                                    "predicted_digit": int(pred_array[t[0], t[1]]) if t[1] < pred_array.shape[1] and pred_array[t[0], t[1]] != -1 else 0,
                                    "confidence": float(t[2]),
                                    "true_digit": int(true_val)
                                } for t in top3 if t[1] < grid.shape[1]
                            ])
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
            
            try:
                response = await requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code != 200:
                    pass
            except requests.RequestException:
                pass
            
            await asyncio.sleep(0.1)
    except HTTPException as e:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Server error")

async def process_batch(input_folder: str, weights: Dict[str, float], return_predictions: bool, output_folder: str, target_num: Optional[int] = None, json_heatmap: Optional[str] = None) -> None:
    if not os.path.exists(input_folder):
        raise HTTPException(status_code=404, detail="Folder does not exist")
    
    os.makedirs(output_folder, exist_ok=True)
    
    tasks: List[asyncio.Task] = []
    for filename in os.listdir(input_folder):
        if filename.endswith(('.json', '.csv', '.xls', '.xlsx')):
            input_filepath = os.path.join(input_folder, filename)
            output_prefix = os.path.join(output_folder, os.path.splitext(filename)[0])
            try:
                tasks.append(
                    asyncio.create_task(
                        process_single_board(
                            input_filepath, weights, return_predictions, output_prefix, target_num, json_heatmap
                        )
                    )
                )
            except Exception:
                continue
    
    if not tasks:
        raise HTTPException(status_code=404, detail="No valid board files found")
    
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
        response = await requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            pass
    except requests.RequestException:
        pass
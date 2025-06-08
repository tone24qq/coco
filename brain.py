import numpy as np
import pandas as pd
import json
import os
import logging
import asyncio
import requests
from typing import List, Tuple, Dict, Any
from analyzer import analyze_board
from fastapi import HTTPException, status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_grid_from_file(filepath: str) -> List[np.ndarray]:
    """
    Loads scratch card grids from a file (JSON, CSV, Excel).

    Args:
        filepath (str): Path to the input file.

    Returns:
        List[np.ndarray]: List of valid grid arrays.

    Raises:
        HTTPException: If file loading fails or no valid grids are found.
    """
    grids: List[np.ndarray] = []
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for grid_data in data:
                    grid = np.array(grid_data, dtype=float)
                    if grid.ndim != 2:
                        logger.warning(f"JSON file {filepath} contains invalid grid, skipping")
                        continue
                    grids.append(grid)
            else:
                grid = np.array(data, dtype=float)
                if grid.ndim == 2:
                    grids.append(grid)
        
        elif ext in ['.csv', '.xls', '.xlsx']:
            if ext == '.csv':
                df = pd.read_csv(filepath, header=None)
                grid = df.to_numpy(dtype=float)
                grids.append(grid)
            else:
                xl = pd.ExcelFile(filepath)
                for sheet_name in xl.sheet_names:
                    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
                    grid = df.to_numpy(dtype=float)
                    grids.append(grid)
        
        cleaned_grids: List[np.ndarray] = []
        for grid in grids:
            grid = np.where(np.isnan(grid) | (grid < 0), -1.0, grid)
            if grid.shape[0] >= 4 and grid.shape[1] >= 4 and grid.shape[0] <= 20 and grid.shape[1] <= 20:
                cleaned_grids.append(grid)
            else:
                logger.warning(f"Grid size {grid.shape} out of 4x4 to 20x20 bounds, skipping")
        
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
    output_format: str
) -> None:
    """
    Saves analysis results to a file in specified format.

    Args:
        scores (np.ndarray): Scores for hidden cells.
        predictions (np.ndarray): Predicted values for the grid.
        best_pos (List[Tuple]): Top 3 predicted positions.
        output_filepath (str): Path to save the output file.
        output_format (str): Format of output ('json', 'csv', 'xls', 'xlsx').

    Raises:
        HTTPException: If saving fails.
    """
    empty_yx = np.argwhere(predictions == -1)
    result = {
        'scores': scores.tolist(),
        'predictions': predictions.tolist(),
        'top3_positions': [{
            'row': int(pos[0]),
            'col': int(pos[1]),
            'confidence': max(float(pos[2]), 0.1),
            'contributions': pos[3]
        } for pos in best_pos],
        'empty_positions': empty_yx.tolist()
    }
    
    try:
        if output_format == 'json':
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        elif output_format == 'csv':
            df = pd.DataFrame({
                'row': empty_yx[:, 0],
                'col': empty_yx[:, 1],
                'score': scores,
                'prediction': predictions[empty_yx[:, 0], empty_yx[:, 1]]
            })
            df.to_csv(output_filepath, index=False)
        
        elif output_format in ['xls', 'xlsx']:
            df = pd.DataFrame({
                'row': empty_yx[:, 0],
                'col': empty_yx[:, 1],
                'score': scores,
                'prediction': predictions[empty_yx[:, 0], empty_yx[:, 1]]
            })
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
    target_num: int = None,
    json_heatmap: str = None
) -> None:
    """
    Processes a single board file and saves results.

    Args:
        filepath (str): Path to input file.
        weights (Dict[str, float]): Module weights.
        return_predictions (bool): Whether to return predictions.
        output_prefix (str): Prefix for output files.
        target_num (int, optional): Target number to locate.
        json_heatmap (str, optional): Path to JSON heatmap directory.

    Raises:
        HTTPException: If processing fails.
    """
    try:
        grids = load_grid_from_file(filepath)
        for idx, grid in enumerate(grids):
            sheet_output_prefix = f"{output_prefix}_sheet{idx+1}"
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            sheet_heatmap_path = os.path.join(json_heatmap, f"{base_name}_sheet{idx+1}.json")
            
            scores, predictions, top3, metrics = analyze_board(
                grid, weights, return_predictions, target_num, sheet_heatmap_path
            )
            
            out_format = os.path.splitext(output_prefix)[1].lower().strip('.')
            if out_format not in ['json', 'csv', 'xls', 'xlsx']:
                sheet_output_prefix += '.json'
                out_format = 'json'
            save_results_to_file(scores, predictions, top3, sheet_output_prefix, out_format)
            
            metrics_filepath = f"{sheet_output_prefix}_metrics.json"
            with open(metrics_filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Sheet {idx+1} processed, results saved to {sheet_output_prefix}")
            
            try:
                response = await requests.get("http://localhost:8000/health", timeout=5)
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
    target_num: int = None,
    json_heatmap: str = None
) -> None:
    """
    Processes multiple board files in a folder.

    Args:
        input_folder (str): Directory containing input files.
        weights (Dict[str, float]): Module weights.
        return_predictions (bool): Whether to return predictions.
        output_folder (str): Directory to save output files.
        target_num (int, optional): Target number to locate.
        json_heatmap (str, optional): Path to JSON heatmap directory.

    Raises:
        HTTPException: If processing fails or no valid files are found.
    """
    if not os.path.exists(input_folder):
        logger.error(f"Input folder {input_folder} does not exist")
        raise HTTPException(status_code=404, detail=f"Folder {input_folder} does not exist")
    
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
            except Exception as e:
                logger.error(f"Failed to schedule task for {input_filepath}: {e}")
                continue
    
    if not tasks:
        logger.error(f"No valid files found in folder {input_folder}")
        raise HTTPException(status_code=404, detail="No valid board files found")
    
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
        response = await requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            logger.warning(f"Health check returned non-200: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Health check failed: {e}")
    
    logger.info(f"Batch processing completed, results saved to {output_folder}")
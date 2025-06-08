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
    grids = []
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for grid_data in data:
                    grid = np.array(grid_data, dtype=float)
                    if grid.ndim != 2:
                        logger.warning(f"JSON 檔案 {filepath} 包含無效盤面，略過")
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
        
        cleaned_grids = []
        for grid in grids:
            grid = np.where(np.isnan(grid) | (grid < 0), -1.0, grid)
            if grid.shape[0] >= 4 and grid.shape[1] >= 4 and grid.shape[0] <= 20 and grid.shape[1] <= 20:
                cleaned_grids.append(grid)
            else:
                logger.warning(f"盤面尺寸 {grid.shape} 超出 4x4 至 20x20，略過")
        
        if not cleaned_grids:
            logger.error(f"檔案 {filepath} 未包含有效盤面")
            raise ValueError("無有效盤面數據")
        
        return cleaned_grids
    
    except Exception as e:
        logger.error(f"載入檔案 {filepath} 失敗: {e}")
        raise HTTPException(status_code=400, detail=f"無法載入盤面: {str(e)}")

def save_results_to_file(scores: np.ndarray, predictions: np.ndarray, best_pos: List[Dict], output_filepath: str, output_format: str):
    empty_yx = np.argwhere(predictions == -1)
    result = {
        'scores': scores.tolist(),
        'predictions': predictions.tolist(),
        'top3_positions': best_pos,
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
        
        logger.info(f"結果已保存至 {output_filepath}")
    
    except Exception as e:
        logger.error(f"保存結果至 {output_filepath} 失敗: {e}")
        raise HTTPException(status_code=500, detail=f"無法保存結果: {str(e)}")

async def process_single_board(filepath: str, weights: dict, return_predictions: bool, output_prefix: str, target_num: int = None, json_heatmap: str = None):
    try:
        grids = load_grid_from_file(filepath)
        for idx, grid in enumerate(grids):
            sheet_output_prefix = f"{output_prefix}_sheet{idx+1}"
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            sheet_heatmap_path = os.path.join(json_heatmap, f"{base_name}_sheet{idx+1}.json")
            
            # 模擬 knowledge_base 和 heatmap_data 為 None，待 app.py 提供
            result = analyze_board(grid, target_num)
            scores = np.array(result.get("confidence", []))
            predictions = np.array([])  # 簡化處理
            top3 = result.get("recommendations", [])
            
            out_format = os.path.splitext(output_prefix)[1].lower().strip('.')
            if out_format not in ['json', 'csv', 'xls', 'xlsx']:
                sheet_output_prefix += '.json'
                out_format = 'json'
            save_results_to_file(scores, predictions, top3, sheet_output_prefix, out_format)
            
            metrics_filepath = f"{sheet_output_prefix}_metrics.json"
            with open(metrics_filepath, 'w', encoding='utf-8') as f:
                json.dump({"metrics": "placeholder"}, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Sheet {idx+1} 處理完成，結果保存至 {sheet_output_prefix}")
            
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code != 200:
                    logger.warning(f"健康檢查回應非 200: {response.status_code}")
            except requests.RequestException as e:
                logger.error(f"健康檢查失敗: {e}")
            
            await asyncio.sleep(0.1)
            
    except HTTPException as e:
        logger.error(f"HTTP 錯誤: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"處理檔案 {filepath} 失敗: {e}")
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

async def process_batch(input_folder: str, weights: dict, return_predictions: bool, output_folder: str, target_num: int = None, json_heatmap: str = None):
    if not os.path.exists(input_folder):
        logger.error(f"輸入資料夾 {input_folder} 不存在")
        raise HTTPException(status_code=404, detail=f"資料夾 {input_folder} 不存在")
    
    os.makedirs(output_folder, exist_ok=True)
    
    tasks = []
    for filename in os.listdir(input_folder):
        if filename.endswith(('.json', '.csv', '.xls', '.xlsx')):
            input_filepath = os.path.join(input_folder, filename)
            output_prefix = os.path.join(output_folder, os.path.splitext(filename)[0])
            try:
                tasks.append(process_single_board(input_filepath, weights, return_predictions, output_prefix, target_num, json_heatmap))
            except Exception as e:
                logger.error(f"安排任務 {input_filepath} 失敗: {e}")
                continue
    
    if not tasks:
        logger.error(f"資料夾 {input_folder} 未找到有效檔案")
        raise HTTPException(status_code=404, detail="未找到有效盤面檔案")
    
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            logger.warning(f"健康檢查回應非 200: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"健康檢查失敗: {e}")
    
    logger.info(f"批次處理完成，結果保存至 {output_folder}")
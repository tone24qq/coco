# app.py（修復後）
from fastapi import FastAPI, File, UploadFile, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import pandas as pd
import json
import os
import logging
import asyncio
from typing import Dict, List, Tuple
from brain import process_single_board, process_batch
from analyzer import analyze_board
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="橘子刮樂分析 API")

class AnalysisRequest(BaseModel):
    grid: List[List[float]] = None
    weights: Dict[str, float] = None
    mode: str = "predict"
    target_num: int = None
    json_heatmap: str = "samples/data/heatmaps"

class HealthCheck(BaseModel):
    status: str = "ok"

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

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {"status": "ok"}

@app.post("/analyze/", status_code=status.HTTP_200_OK)
async def analyze_grid(request: AnalysisRequest, background_tasks: BackgroundTasks):
    try:
        if request.grid is None:
            raise HTTPException(status_code=400, detail="未提供盤面數據")
        
        grid = np.array(request.grid, dtype=float)
        if grid.ndim != 2 or grid.shape[0] > 20 or grid.shape[1] > 20:
            raise HTTPException(status_code=400, detail="盤面尺寸無效，必須為 4x5 至 20x20")
        
        weights = request.weights if request.weights else DEFAULT_WEIGHTS
        return_predictions = (request.mode == "predict")
        
        json_heatmap_path = os.path.join(request.json_heatmap, "temp_grid.json")
        os.makedirs(request.json_heatmap, exist_ok=True)
        
        scores, predictions, top3, metrics = analyze_board(
            grid, weights, return_predictions, request.target_num, json_heatmap_path
        )
        
        result = {
            "scores": scores.tolist(),
            "predictions": predictions.tolist(),
            "top3_positions": [{
                "row": pos[0],
                "col": pos[1],
                "confidence": max(float(pos[2]), 0.1),
                "contributions": pos[3]
            } for pos in top3],
            "metrics": metrics
        }
        
        output_path = "samples/output/api_result.json"
        background_tasks.add_task(
            save_results_to_file, scores, predictions, top3, output_path, "json"
        )
        
        return JSONResponse(content=result, status_code=200)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"分析盤面失敗: {e}")
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@app.post("/upload/", status_code=status.HTTP_200_OK)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.json', '.csv', '.xls', '.xlsx')):
            raise HTTPException(status_code=400, detail="不支援的檔案格式")
        
        input_path = f"samples/data/{file.filename}"
        os.makedirs("samples/data", exist_ok=True)
        
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        output_prefix = f"samples/output/{os.path.splitext(file.filename)[0]}"
        weights = DEFAULT_WEIGHTS
        return_predictions = True
        json_heatmap = "samples/data/heatmaps"
        
        background_tasks.add_task(
            process_single_board, input_path, weights, return_predictions, output_prefix, None, json_heatmap
        )
        
        return JSONResponse(
            content={"message": f"檔案 {file.filename} 已上傳，處理中", "output_path": output_prefix},
            status_code=200
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"上傳檔案失敗: {e}")
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@app.post("/batch/", status_code=status.HTTP_200_OK)
async def batch_process(input_folder: str, background_tasks: BackgroundTasks):
    try:
        if not os.path.exists(input_folder):
            raise HTTPException(status_code=404, detail=f"資料夾 {input_folder} 不存在")
        
        output_folder = f"samples/output/batch_{os.path.basename(input_folder)}"
        weights = DEFAULT_WEIGHTS
        return_predictions = True
        json_heatmap = "samples/data/heatmaps"
        
        background_tasks.add_task(
            process_batch, input_folder, weights, return_predictions, output_folder, None, json_heatmap
        )
        
        return JSONResponse(
            content={"message": f"批次處理已開始，結果將保存至 {output_folder}"},
            status_code=200
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"批次處理失敗: {e}")
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

def save_results_to_file(scores: np.ndarray, predictions: np.ndarray, best_pos: List[Tuple], output_filepath: str, output_format: str):
    from brain import save_results_to_file as brain_save
    brain_save(scores, predictions, best_pos, output_filepath, output_format)

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, full_path: str):
    logger.debug(f"Catch-all for path: {request.method} {full_path}")
    return JSONResponse(status_code=200, content={"status": "running"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
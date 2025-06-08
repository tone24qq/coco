from fastapi import FastAPI, File, UploadFile, HTTPException, status, BackgroundTasks, Request, Form
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import pandas as pd
import json
import os
import logging
import asyncio
import glob
from typing import Dict, List, Tuple
from brain import process_single_board, process_batch, load_grid_from_file
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
    json_heatmap: str = "samples/data/json"

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
    "analyze_number_patterns": 0.05,
    "detect_diagonal_pattern": 0.05,
    "compute_spatial_correlation": 0.05,
    "interference_penalty": 0.05
}

# 全域熱力圖快取與知識庫
HEATMAP_CACHE: Dict[str, np.ndarray] = {}
math_algo_kb = []

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "samples", "data")
json_paths = glob.glob(os.path.join(DATA_DIR, "*.json"))

# 分類 JSON 路徑
kb_path = os.path.join(DATA_DIR, "math_algo_kb.json")
heatmap_paths = [p for p in json_paths if os.path.basename(p).startswith("heatmap_")]

# 讀取知識庫
try:
    with open(kb_path, encoding="utf-8") as f:
        math_algo_kb = json.load(f)["concepts"]
    logger.info(f"Loaded {len(math_algo_kb)} KB concepts")
except FileNotFoundError:
    math_algo_kb = []
    logger.warning(f"Warning: knowledge base not found at {kb_path}")

# 讀取熱力圖
heatmaps = {}
for hp in heatmap_paths:
    name = os.path.splitext(os.path.basename(hp))[0]
    with open(hp, encoding="utf-8") as f:
        heatmaps[name] = json.load(f)

@app.on_event("startup")
def load_heatmap_cache():
    pass  # 熱力圖已在啟動時預載

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {"status": "ok"}

@app.post("/analyze/")
async def analyze(
    file: UploadFile = File(...),
    mode: str = Form("heatmap"),
    target_num: int = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        if not file.filename.endswith(('.json', '.csv', '.xls', '.xlsx')):
            raise HTTPException(status_code=400, detail="不支援的檔案格式")
        
        input_path = os.path.join(DATA_DIR, file.filename)
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        grids = load_grid_from_file(input_path)
        if not grids:
            raise HTTPException(status_code=400, detail="無有效盤面數據")
        
        grid = grids[0]  # 處理第一個盤面
        result = analyze_board(grid, target_num, knowledge_base=math_algo_kb, heatmap_data=heatmaps)
        
        output_path = os.path.join("samples/output", f"api_result_{file.filename}.json")
        background_tasks.add_task(save_results_to_file, np.array(result.get("confidence", [])), np.array([]), result.get("recommendations", []), output_path, "json")
        
        return JSONResponse(content=result, status_code=200)
    
    except HTTPException as e:
        logger.error(f"HTTP 錯誤: {e.detail}")
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail, "recommendations": [{"row": 0, "col": 0, "confidence": 0.1, "weights": {}}]}
        )
    except Exception as e:
        logger.error(f"分析盤面失敗: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"伺服器錯誤: {str(e)}", "recommendations": [{"row": 0, "col": 0, "confidence": 0.1, "weights": {}}]}
        )

@app.post("/upload/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.json', '.csv', '.xls', '.xlsx')):
            raise HTTPException(status_code=400, detail="不支援的檔案格式")
        
        input_path = os.path.join(DATA_DIR, file.filename)
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        output_prefix = os.path.join("samples/output", os.path.splitext(file.filename)[0])
        weights = DEFAULT_WEIGHTS
        return_predictions = True
        json_heatmap = "samples/data/json"
        
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

@app.post("/batch/")
async def batch_process(input_folder: str, background_tasks: BackgroundTasks):
    try:
        if not os.path.exists(input_folder):
            raise HTTPException(status_code=404, detail=f"資料夾 {input_folder} 不存在")
        
        output_folder = f"samples/output/batch_{os.path.basename(input_folder)}"
        weights = DEFAULT_WEIGHTS
        return_predictions = True
        json_heatmap = "samples/data/json"
        
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

def save_results_to_file(scores: np.ndarray, predictions: np.ndarray, best_pos: List[Dict], output_filepath: str, output_format: str):
    from brain import save_results_to_file as brain_save
    brain_save(scores, predictions, best_pos, output_filepath, output_format)

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, full_path: str):
    logger.debug(f"Catch-all for path: {request.method} {full_path}")
    return JSONResponse(status_code=200, content={"status": "running"})
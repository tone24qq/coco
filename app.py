import os
import json
import zipfile
import logging
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse

from modules import ScratchSolver
from analyzer import analyze_board

# 日誌設定
logging.basicConfig(
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# FastAPI App & Solver
app = FastAPI()
solver = ScratchSolver()

# Startup: 批次前置處理
@app.on_event("startup")
async def startup_process_samples():
    default_weights = {
        "focus": 0.2,
        "skip": 0.15,
        "diff": 0.15,
        "mirror": 0.2,
        "conn": 0.15,
        "tail": 0.15,
        "constraint": 0.1,
        "tensor": 0.1,
        "pattern": 0.1,
        "json": 0.1
    }
    return_predictions = False

    folder_path = "samples/data/"
    output_folder = "samples/output/"
    os.makedirs(output_folder, exist_ok=True)

    logger.info(f"Startup: 開始掃描資料夾 {folder_path}")
    if not os.path.exists(folder_path):
        logger.error(f"Startup: 資料夾 {folder_path} 不存在，跳過啟動處理")
        return

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".json", ".csv", ".xls", ".xlsx"]:
            logger.info(f"Startup: 跳過不支援的檔案 {filename}")
            continue

        filepath = os.path.join(folder_path, filename)
        try:
            grids = load_file_content(filepath)
        except Exception as e:
            logger.error(f"Startup: 讀取 {filename} 失敗：{e}")
            continue

        for idx, grid in enumerate(grids):
            try:
                heatmap, _, _ = analyze_board(grid, default_weights, return_predictions)
            except Exception as e:
                logger.error(f"Startup: 處理 {filename} 第 {idx+1} 張 grid 失敗：{e}")
                continue

            base = os.path.splitext(filename)[0]
            out_name = f"{base}_sheet{idx+1}_heatmap.json" if ext in [".xls", ".xlsx"] else f"{base}_heatmap.json"
            out_path = os.path.join(output_folder, out_name)

            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"heatmap": heatmap.tolist()}, f, ensure_ascii=False, indent=2)
                logger.info(f"Startup: 已存 {filename} (sheet {idx+1}) 的熱力圖到 {out_path}")
            except Exception as e:
                logger.error(f"Startup: 存檔 {out_path} 失敗：{e}")

    logger.info("Startup: Samples 資料夾熱力圖前置處理完成")

# 核心處理函式
def process_grid(
    grid: np.ndarray,
    weights: dict,
    return_predictions: bool,
    target_num: int = None,
    json_heatmap: str = None
) -> dict:
    if grid.size == 0 or grid.shape[0] > 20 or grid.shape[1] > 20:
        return {"error": "網格為空或超過 20x20 限制", "timestamp": datetime.now().isoformat()}

    try:
        final_score, final_pred, best_pos = analyze_board(
            grid, weights, return_predictions, target_num, json_heatmap
        )
    except ValueError as e:
        logger.error(f"盤面分析失敗：{str(e)}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

    result = {
        "heatmap": final_score.tolist() if final_score is not None else [],
        "timestamp": datetime.now().isoformat()
    }
    if return_predictions and final_pred is not None:
        result["prediction"] = final_pred.tolist()
        if best_pos:
            if isinstance(best_pos, dict) and "error" in best_pos:
                result["error"] = best_pos["error"]
            else:
                result["best_positions"] = [
                    {
                        "coords": [pos[0], pos[1]],
                        "confidence": float(pos[2]),
                        "reasoning": pos[3]
                    } for pos in best_pos[:3]  # 返回Top3
                ]
    return result

def parse_weights(weights: str) -> dict:
    if weights:
        try:
            return json.loads(weights)
        except json.JSONDecodeError:
            logger.error("無效的權重 JSON 格式")
            raise ValueError("無效的權重 JSON 格式")
    return {
        "focus": 0.2,
        "skip": 0.15,
        "diff": 0.15,
        "mirror": 0.2,
        "conn": 0.15,
        "tail": 0.15,
        "constraint": 0.1,
        "tensor": 0.1,
        "pattern": 0.1,
        "json": 0.1
    }

def load_file_content(filepath: str) -> list:
    logger.info(f"Processing file: {filepath}")
    ext = os.path.splitext(filepath)[1].lower()
    grids = []

    if ext == ".json":
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        grids = [np.array(data)]

    elif ext == ".csv":
        df = pd.read_csv(filepath, header=None, dtype=str).fillna("")
        cleaned = [
            [int(c.replace('O','0').replace('I','1')) if c.isdigit() else -1 for c in row]
            for row in df.values
        ]
        grids = [np.array(cleaned)]

    else:  # .xls, .xlsx
        xls = pd.ExcelFile(filepath)
        for sheet in xls.sheet_names:
            logger.info(f"Processing sheet: {sheet}")
            df = pd.read_excel(filepath, sheet_name=sheet, header=None, dtype=str).fillna("")
            cleaned = [
                [int(c.replace('O','0').replace('I','1')) if c.isdigit() else -1 for c in row]
                for row in df.values
            ]
            grids.append(np.array(cleaned))

    return grids

# HTTP 路由
@app.post("/analyze/")
async def analyze(
    file: UploadFile = File(...),
    weights: str = Form(None),
    mode: str = Form("predict"),
    target_num: int = Form(None),
    json_heatmap: str = Form(None)
):
    filename = file.filename.lower()
    logger.info(f"Analyzing uploaded file: {filename}")
    if not filename.endswith((".xls", ".xlsx", ".json", ".csv")):
        return JSONResponse(status_code=400, content={"error": "不支援的檔案格式
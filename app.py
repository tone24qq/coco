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
        return JSONResponse(status_code=400, content={"error": "不支援的檔案格式", "timestamp": datetime.now().isoformat()})

    content = await file.read()
    grids = []

    if filename.endswith(".json"):
        try:
            data = json.loads(content.decode("utf-8"))
            grids = [np.array(data)]
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "無效的 JSON 格式", "timestamp": datetime.now().isoformat()})

    elif filename.endswith(".csv"):
        df = pd.read_csv(BytesIO(content), header=None, dtype=str).fillna("")
        cleaned = [
            [int(c.replace('O','0').replace('I','1')) if c.isdigit() else -1 for c in row]
            for row in df.values
        ]
        grids = [np.array(cleaned)]

    else:  # .xls, .xlsx
        try:
            xls = pd.ExcelFile(BytesIO(content))
            for sheet_name in xls.sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")
                df = pd.read_excel(BytesIO(content), sheet_name=sheet_name, header=None, dtype=str).fillna("")
                cleaned = [
                    [int(c.replace('O','0').replace('I','1')) if c.isdigit() else -1 for c in row]
                    for row in df.values
                ]
                grids.append(np.array(cleaned))
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Excel 讀取失敗: {str(e)}", "timestamp": datetime.now().isoformat()})

    try:
        w_dict = parse_weights(weights)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e), "timestamp": datetime.now().isoformat()})

    return_predictions = (mode == "predict")
    results = []
    for idx, grid in enumerate(grids):
        result = process_grid(grid, w_dict, return_predictions, target_num, json_heatmap)
        result["sheet"] = idx + 1
        results.append(result)

    return JSONResponse(content={"results": results})

@app.post("/analyze-batch/")
async def analyze_batch(
    file: UploadFile = File(...),
    weights: str = Form(None),
    mode: str = Form("predict"),
    target_num: int = Form(None),
    json_heatmap: str = Form(None)
):
    filename = file.filename.lower()
    logger.info(f"Analyzing batch file: {filename}")
    if not filename.endswith(".zip"):
        return JSONResponse(status_code=400, content={"error": "請上傳 ZIP 檔案", "timestamp": datetime.now().isoformat()})

    content = await file.read()
    results = []
    try:
        w_dict = parse_weights(weights)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e), "timestamp": datetime.now().isoformat()})

    return_predictions = (mode == "predict")
    try:
        with zipfile.ZipFile(BytesIO(content)) as z:
            for zip_info in z.infolist():
                name = zip_info.filename
                if name.endswith((".xls", ".xlsx")):
                    with z.open(zip_info) as f:
                        xls = pd.ExcelFile(BytesIO(f.read()))
                        file_results = {"filename": name, "sheets": []}
                        for sheet in xls.sheet_names:
                            logger.info(f"Processing sheet: {sheet}")
                            df = pd.read_excel(BytesIO(f.read()), sheet_name=sheet, header=None, dtype=str).fillna("")
                            cleaned = [
                                [int(c.replace('O','0').replace('I','1')) if c.isdigit() else -1 for c in row]
                                for row in df.values
                            ]
                            grid = np.array(cleaned)
                            result = process_grid(grid, w_dict, return_predictions, target_num, json_heatmap)
                            result["sheet"] = sheet
                            file_results["sheets"].append(result)
                        results.append(file_results)
                elif name.endswith(".json"):
                    with z.open(zip_info) as f:
                        try:
                            data = json.load(f)
                            grid = np.array(data)
                        except json.JSONDecodeError:
                            results.append({"filename": name, "error": "無效的 JSON 格式", "timestamp": datetime.now().isoformat()})
                            continue
                        file_results = {"filename": name, "sheets": []}
                        result = process_grid(grid, w_dict, return_predictions, target_num, json_heatmap)
                        result["sheet"] = "data"
                        file_results["sheets"].append(result)
                        results.append(file_results)
                elif name.endswith(".csv"):
                    with z.open(zip_info) as f:
                        df = pd.read_csv(BytesIO(f.read()), header=None, dtype=str).fillna("")
                        cleaned = [
                            [int(c.replace('O','0').replace('I','1')) if c.isdigit() else -1 for c in row]
                            for row in df.values
                        ]
                        grid = np.array(cleaned)
                        file_results = {"filename": name, "sheets": []}
                        result = process_grid(grid, w_dict, return_predictions, target_num, json_heatmap)
                        result["sheet"] = "data"
                        file_results["sheets"].append(result)
                        results.append(file_results)
    except zipfile.BadZipFile:
        return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔案", "timestamp": datetime.now().isoformat()})

    return JSONResponse(content={"results": results})

@app.post("/analyze-folder/")
async def analyze_folder(
    weights: str = Form(None),
    mode: str = Form("predict"),
    target_num: int = Form(None),
    json_heatmap: str = Form(None)
):
    folder_path = "samples/data/"
    logger.info(f"Scanning folder: {folder_path}")
    if not os.path.exists(folder_path):
        logger.error(f"Folder {folder_path} does not exist")
        return JSONResponse(content={"error": f"資料夾 {folder_path} 不存在", "timestamp": datetime.now().isoformat()})

    files = os.listdir(folder_path)
    logger.info(f"Found {len(files)} files in {folder_path}: {files}")
    if not files:
        logger.warning(f"No files found in {folder_path}")
        return JSONResponse(content={"content": f"資料夾 {folder_path} 為空", "timestamp": datetime.now().isoformat()})

    try:
        w_dict = parse_weights(weights)
    except Exception as e:
        logger.error(f"Weight parsing error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": str(e), "timestamp": datetime.now().isoformat()})

    return_predictions = (mode == "predict")
    results = []
    for idx, filename in enumerate(files):
        filepath = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".json", ".csv", ".xls", ".xlsx"]:
            logger.info(f"Skipping non-supported file: {filename}")
            continue
        try:
            grids = load_file_content(filepath)
            file_results = {"filename": filename, "sheets": []}
            for sidx, grid in enumerate(grids):
                logger.info(f"Processing grid {sidx+1} in {filename}")
                result = process_grid(grid, w_dict, return_predictions, target_num, json_heatmap)
                result["sheet"] = f"Sheet{sidx+1}" if ext in [".xls", ".xlsx"] else "data"
                file_results["sheets"].append(result)
            results.append(file_results)
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            results.append({"filename": filename, "error": f"檔案處理失敗: {str(e)}", "timestamp": datetime.now().isoformat()})

    return JSONResponse(content={"results": results})

# 根路由
@app.get("/")
async def root():
    return JSONResponse(status_code=200, content={"status": "running", "timestamp": datetime.now().isoformat()})

# Catch-all 路由
@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
)
async def catch_all(request: Request, full_path: str):
    logger.debug(f"Catch-all for path: {request.method} {full_path}")
    return JSONResponse(status_code=200, content={"status": "running", "timestamp": datetime.now().isoformat()})
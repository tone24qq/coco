from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import numpy as np
import json
import pandas as pd
from io import BytesIO
import os
import zipfile
import logging

from modules import ScratchSolver
from analyzer import analyze_board

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
solver = ScratchSolver()

@app.on_event("startup")
async def startup_process_samples():
    """
    服務一啟動，就走遍 samples/data/ 資料夾，把所有支援的檔案讀進來，
    針對每個 grid 計算熱力分數，並把結果存到 samples/output/ 底下（檔名：原檔名+_heatmap.json）。
    """
    # 預設權重（和 analyze_folder 一樣）
    default_weights = {
        "focus": 0.2,
        "skip": 0.15,
        "diff": 0.15,
        "mirror": 0.2,
        "conn": 0.15,
        "tail": 0.15
    }
    return_predictions = False  # 只要熱力圖，不要預測值

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
            # 用跟 analyze_folder 一樣的 load_file_content 去把所有工作表都讀成 ndarray list
            grids = load_file_content(filepath)
        except Exception as e:
            logger.error(f"Startup: 讀取 {filename} 失敗：{e}")
            continue

        for idx, grid in enumerate(grids):
            # 處理每一張 grid，拿到熱力分數
            try:
                heatmap, _ = analyze_board(grid, default_weights, return_predictions)
            except Exception as e:
                logger.error(f"Startup: 處理 {filename} 第 {idx+1} 張 grid 失敗：{e}")
                continue

            # 把 heatmap 存成 JSON，檔名：原本檔名（不含副檔名）_sheet{idx+1}_heatmap.json
            base = os.path.splitext(filename)[0]
            if ext in [".xls", ".xlsx"]:
                out_name = f"{base}_sheet{idx+1}_heatmap.json"
            else:
                out_name = f"{base}_heatmap.json"
            out_path = os.path.join(output_folder, out_name)

            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"heatmap": heatmap.tolist()}, f, ensure_ascii=False, indent=2)
                logger.info(f"Startup: 已存 {filename} (sheet {idx+1}) 的熱力圖到 {out_path}")
            except Exception as e:
                logger.error(f"Startup: 存檔 {out_path} 失敗：{e}")

    logger.info("Startup: Samples 資料夾熱力圖前置處理完成")


def process_grid(grid: np.ndarray, weights: dict, return_predictions: bool) -> dict:
    if grid.size == 0 or grid.shape[0] > 20 or grid.shape[1] > 20:
        return {"error": "網格為空或超過 20x20 限制"}
    final_score, final_pred = analyze_board(grid, weights, return_predictions)
    result = {"heatmap": final_score.tolist()}
    if return_predictions and final_pred is not None:
        result["prediction"] = final_pred.tolist()
    return result

def parse_weights(weights: str) -> dict:
    if weights:
        try:
            return json.loads(weights)
        except json.JSONDecodeError:
            raise ValueError("無效的權重 JSON 格式")
    return {
        "focus": 0.2,
        "skip": 0.15,
        "diff": 0.15,
        "mirror": 0.2,
        "conn": 0.15,
        "tail": 0.15
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
        df = pd.read_csv(filepath, header=None, dtype=str)
        df = df.fillna("")
        cleaned_data = []
        for row in df.values:
            cleaned_row = [int(cell.replace('O', '0').replace('I', '1')) if cell.isdigit() else -1 for cell in row]
            cleaned_data.append(cleaned_row)
        grids = [np.array(cleaned_data)]
    elif ext in [".xls", ".xlsx"]:
        xls = pd.ExcelFile(filepath)
        for sheet_name in xls.sheet_names:
            logger.info(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=None, dtype=str)
            df = df.fillna("")
            cleaned_data = []
            for row in df.values:
                cleaned_row = [int(cell.replace('O', '0').replace('I', '1')) if cell.isdigit() else -1 for cell in row]
                cleaned_data.append(cleaned_row)
            grids.append(np.array(cleaned_data))
    return grids

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...),
                 weights: str = Form(None),
                 mode: str = Form("heatmap")):
    filename = file.filename.lower()
    logger.info(f"Analyzing uploaded file: {filename}")
    if not filename.endswith((".xls", ".xlsx", ".json", ".csv")):
        return JSONResponse(status_code=400, content={"error": "不支援的檔案格式"})

    content = await file.read()
    grids = []
    if filename.endswith(".json"):
        try:
            data = json.loads(content.decode("utf-8"))
            grids = [np.array(data)]
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "無效的 JSON 格式"})
    elif filename.endswith(".csv"):
        df = pd.read_csv(BytesIO(content), header=None, dtype=str)
        df = df.fillna("")
        cleaned_data = []
        for row in df.values:
            cleaned_row = [int(cell.replace('O', '0').replace('I', '1')) if cell.isdigit() else -1 for cell in row]
            cleaned_data.append(cleaned_row)
        grids = [np.array(cleaned_data)]
    elif filename.endswith((".xls", ".xlsx")):
        try:
            xls = pd.ExcelFile(BytesIO(content))
            for sheet_name in xls.sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")
                df = pd.read_excel(BytesIO(content), sheet_name=sheet_name, header=None, dtype=str)
                df = df.fillna("")
                cleaned_data = []
                for row in df.values:
                    cleaned_row = [int(cell.replace('O', '0').replace('I', '1')) if cell.isdigit() else -1 for cell in row]
                    cleaned_data.append(cleaned_row)
                grids.append(np.array(cleaned_data))
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Excel 讀取失敗: {str(e)}"})

    try:
        w_dict = parse_weights(weights)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return_predictions = (mode == "predict")
    results = []
    for idx, grid in enumerate(grids):
        result = process_grid(grid, w_dict, return_predictions)
        result["sheet"] = idx + 1
        results.append(result)

    return JSONResponse(content={"results": results})

@app.post("/analyze-batch/")
async def analyze_batch(file: UploadFile = File(...),
                       weights: str = Form(None),
                       mode: str = Form("heatmap")):
    filename = file.filename.lower()
    logger.info(f"Analyzing batch file: {filename}")
    if not filename.endswith(".zip"):
        return JSONResponse(status_code=400, content={"error": "請上傳 ZIP 檔案"})

    content = await file.read()
    results = []
    try:
        w_dict = parse_weights(weights)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return_predictions = (mode == "predict")
    try:
        with zipfile.ZipFile(BytesIO(content)) as z:
            for zip_info in z.infolist():
                if zip_info.filename.endswith((".xls", ".xlsx")):
                    with z.open(zip_info) as f:
                        xls = pd.ExcelFile(BytesIO(f.read()))
                        file_results = {"filename": zip_info.filename, "sheets": []}
                        for sheet_name in xls.sheet_names:
                            logger.info(f"Processing sheet: {sheet_name}")
                            df = pd.read_excel(BytesIO(f.read()), sheet_name=sheet_name, header=None, dtype=str)
                            df = df.fillna("")
                            cleaned_data = []
                            for row in df.values:
                                cleaned_row = [int(cell.replace('O', '0').replace('I', '1')) if cell.isdigit() else -1 for cell in row]
                                cleaned_data.append(cleaned_row)
                            grid = np.array(cleaned_data)
                            result = process_grid(grid, w_dict, return_predictions)
                            result["sheet"] = sheet_name
                            file_results["sheets"].append(result)
                        results.append(file_results)
                elif zip_info.filename.endswith((".json", ".csv")):
                    with z.open(zip_info) as f:
                        if zip_info.filename.endswith(".json"):
                            try:
                                data = json.load(f)
                                grid = np.array(data)
                            except json.JSONDecodeError:
                                results.append({"filename": zip_info.filename, "error": "無效的 JSON 格式"})
                                continue
                        else:
                            df = pd.read_csv(BytesIO(f.read()), header=None, dtype=str)
                            df = df.fillna("")
                            cleaned_data = []
                            for row in df.values:
                                cleaned_row = [int(cell.replace('O', '0').replace('I', '1')) if cell.isdigit() else -1 for cell in row]
                                cleaned_data.append(cleaned_row)
                            grid = np.array(cleaned_data)
                        file_results = {"filename": zip_info.filename, "sheets": []}
                        result = process_grid(grid, w_dict, return_predictions)
                        result["sheet"] = "data"
                        file_results["sheets"].append(result)
                        results.append(file_results)
    except zipfile.BadZipFile:
        return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔案"})

    return JSONResponse(content={"results": results})

@app.post("/analyze-folder/")
async def analyze_folder(weights: str = Form(None), mode: str = Form("heatmap")):
    folder_path = "samples/data/"
    logger.info(f"Scanning folder: {folder_path}")
    if not os.path.exists(folder_path):
        logger.error(f"Folder {folder_path} does not exist")
        return JSONResponse(content={"error": f"資料夾 {folder_path} 不存在"})

    files = os.listdir(folder_path)
    logger.info(f"Found {len(files)} files in {folder_path}: {files}")
    if not files:
        logger.warning(f"No files found in {folder_path}")
        return JSONResponse(content={"content": f"資料夾 {folder_path} 為空"})

    try:
        w_dict = parse_weights(weights)
    except Exception as e:
        logger.error(f"Weight parsing error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": str(e)})

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
            for idx, grid in enumerate(grids):
                logger.info(f"Processing grid {idx+1} in {filename}")
                result = process_grid(grid, w_dict, return_predictions)
                result["sheet"] = f"Sheet{idx+1}" if ext in [".xls", ".xlsx"] else "data"
                file_results["sheets"].append(result)
            results.append(file_results)
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            results.append({"filename": filename, "error": f"檔案處理失敗: {str(e)}"})

    return JSONResponse(content={"results": results})

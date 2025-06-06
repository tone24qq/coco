from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import json
import pandas as pd
from io import BytesIO
import os
import zipfile
from modules import ScratchSolver
from analyzer import analyze_board

app = FastAPI()
solver = ScratchSolver()

@app.get("/")
async def root():
    return {"status": "API is running", "endpoint": "POST /analyze/ or /analyze-batch/ for Excel analysis"}

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

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...),
                 weights: str = Form(None),
                 mode: str = Form("heatmap")):
    filename = file.filename.lower()
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
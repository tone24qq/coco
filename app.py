from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import json
import pandas as pd
from io import BytesIO
from modules import ScratchSolver
from analyzer import analyze_board

app = FastAPI()
solver = ScratchSolver()

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...),
                 weights: str = Form(None),
                 mode: str = Form("heatmap")):
    # 驗證檔案格式
    filename = file.filename.lower()
    if not filename.endswith((".xls", ".xlsx", ".json", ".csv")):
        return JSONResponse(status_code=400, content={"error": "不支援的檔案格式"})

    # 讀取檔案內容
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

    # 解析權重
    if weights:
        try:
            w_dict = json.loads(weights)
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "無效的權重 JSON 格式"})
    else:
        w_dict = {
            "focus": 0.2,
            "skip": 0.15,
            "diff": 0.15,
            "mirror": 0.2,
            "conn": 0.15,
            "tail": 0.15
        }

    # 處理每個網格
    return_predictions = (mode == "predict")
    results = []
    for idx, grid in enumerate(grids):
        if grid.size == 0 or grid.shape[0] > 20 or grid.shape[1] > 20:
            results.append({"sheet": idx + 1, "error": "網格為空或超過 20x20 限制"})
            continue
        final_score, final_pred = analyze_board(grid, w_dict, return_predictions)
        result = {"sheet": idx + 1, "heatmap": final_score.tolist()}
        if return_predictions and final_pred is not None:
            result["prediction"] = final_pred.tolist()
        results.append(result)

    return JSONResponse(content={"results": results})
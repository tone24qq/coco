# main14.py
"""
main14.py：FastAPI 應用，提供 /predict Endpoint，並在背景不斷更新樣本權重與做防禦檢查。
"""

import asyncio
import os
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import numpy as np

from analyzer11 import (
    collect_all_scores,
    normalize_tensor,
    fuse_scores,
    get_topk_positions,
    get_weights_for_shape,
    maybe_reload_memory  # 建議在 analyzer11.py 裡實作此函式，以檔案變動方式觸發重載
)

# 設定日誌輸出
logging.basicConfig(
    filename="predict.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 最大允許 grid 大小
MAX_ROWS = 50
MAX_COLS = 50

class PredictRequest(BaseModel):
    grid: list[list[int]]
    target: int

class PredictResponse(BaseModel):
    predictions: list[dict]
    error: str | None = None

@app.on_event("startup")
async def startup_event():
    """
    啟動時觸發：每隔 60 秒檢查一次 memory_data 資料夾是否有更新，
    若有則自動 reload MEMORY_SAMPLES 並更新權重。
    """
    async def _periodic_reload():
        while True:
            try:
                maybe_reload_memory()  # analyzer11.py 裡實作檔案變動檢查與重載
            except Exception as e:
                logger.error(f"Background reload 發生錯誤：{e}")
            await asyncio.sleep(60)  # 每 60 秒檢查一次

    # 啟動背景任務
    asyncio.create_task(_periodic_reload())


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """
    捕捉 Pydantic 驗證錯誤，回 400 並回傳欄位錯誤訊息。
    """
    return JSONResponse(
        status_code=400,
        content={"predictions": [], "error": "請求參數驗證失敗: " + str(exc)}
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    /predict Endpoint：
    1. 驗證請求內容
    2. 動態載入最新權重（已在背景自動更新）
    3. 收集各模組分數、正規化、融合（根據當前卡片尺寸選權重）
    4. 取 Top-3 並回傳
    """
    # --- 1) 驗證 grid 是否為非空的矩形且尺寸 <= 50×50 ---
    grid = req.grid
    target = req.target

    rows = len(grid)
    if rows == 0:
        logger.error("收到空 Grid")
        raise HTTPException(status_code=400, detail="Grid 不能為空")
    cols = len(grid[0])
    if rows > MAX_ROWS or cols > MAX_COLS:
        logger.error(f"Grid 大小 {rows}x{cols} 超過 {MAX_ROWS}x{MAX_COLS}")
        raise HTTPException(status_code=400, detail=f"Grid 大小超過 {MAX_ROWS}×{MAX_COLS} 限制")
    for row in grid:
        if len(row) != cols:
            logger.error("Grid 不是矩形")
            raise HTTPException(status_code=400, detail="Grid 必須為矩形")

    # 驗證 cell 值：必須是 int，且要么為 -1，要么 > 0，不可重複
    seen = set()
    for r in range(rows):
        row = grid[r]
        for c in range(cols):
            val = row[c]
            if not isinstance(val, int):
                logger.error("Grid 含非整數值")
                raise HTTPException(status_code=400, detail="Grid 值必須是整數")
            if val != -1 and val <= 0:
                logger.error("Grid 含非正值或非 -1")
                raise HTTPException(status_code=400, detail="Grid 值必須為正整數或 -1")
            if val != -1:
                if val in seen:
                    logger.error("Grid 含重複值")
                    raise HTTPException(status_code=400, detail="Grid 內含重複值")
                seen.add(val)

    # 驗證 target
    if not isinstance(target, int):
        logger.error("Target 不是整數")
        raise HTTPException(status_code=400, detail="Target 必須是整數")
    if target <= 0:
        logger.error("Target 非正整數")
        raise HTTPException(status_code=400, detail="Target 必須為正整數")
    if target in seen:
        logger.error("Target 已存在於 Grid 中")
        raise HTTPException(status_code=400, detail="Target 已存在於 Grid 中")

    # --- 2) 執行推論流程 ---
    try:
        arr = np.array(grid, dtype=int)
        blank_count = int(np.sum(arr == -1))
        logger.info(f"[Predict] target={target}, shape={rows}x{cols}, blanks={blank_count}")

        # 動態取權重：根據當前 shape 先不重新計算，因為背景任務已在定期 reload
        # collect_all_scores 可改為非同步函式，但目前保持同步運算
        tensor = collect_all_scores(arr, request_id="API")
        if tensor.size == 0:
            logger.error("No modules returned any scores")
            raise HTTPException(status_code=500, detail="No scoring modules available")

        tensor_norm = normalize_tensor(tensor, method="minmax")

        # 依照 grid.shape 取得最佳權重；如果沒有，就回傳 global 或等權
        weights_dict = get_weights_for_shape((rows, cols))
        if weights_dict:
            # 模組名稱順序需與 new_module3.REGISTERED_MODULES_BRAIN 保持一致
            from new_module3 import REGISTERED_MODULES_BRAIN
            module_names = list(REGISTERED_MODULES_BRAIN.keys())
            weights_list = [weights_dict.get(name, 0.0) for name in module_names]
            fused = fuse_scores(tensor_norm, weights=weights_list)
        else:
            fused = fuse_scores(tensor_norm, weights=None)

        topk = get_topk_positions(fused, arr, k=3)
        predictions = []
        total_fused = float(np.nansum(fused[arr == -1])) if np.any(arr == -1) else 1.0
        for (r, c), score in topk:
            confidence = float(score / total_fused) if total_fused > 0 else 0.0
            predictions.append({
                "row": r + 1,      # 1-based index
                "col": c + 1,
                "confidence": round(confidence, 6)
            })

        logger.info(f"[Predict] Top-3 => {predictions}")
        return {"predictions": predictions, "error": None}

    except HTTPException as e:
        # 已知錯誤直接回傳
        raise e
    except Exception as e:
        # 統一記錄例外，避免回傳機密
        logger.exception("未知錯誤於 /predict")
        raise HTTPException(status_code=500, detail="Internal server error")
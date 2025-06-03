# main14.py
"""
main14.py：FastAPI 應用，提供 /predict Endpoint，
並整合「模組融合 + 歷史先驗」邏輯。請複製以下內容，直接覆蓋原檔。
"""

import asyncio
import os
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
    get_target_prior,      # ← 確保 analyzer11.py 已實作此函式
    maybe_reload_memory
)
from new_module3 import REGISTERED_MODULES_BRAIN

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
    啟動時建立背景任務：每隔 60 秒檢查一次 memory_data 資料夾，如有更新則 reload
    """
    async def _periodic_reload():
        while True:
            try:
                maybe_reload_memory()  # 每次檢查，若有新樣本自動重載 & 重算權重
            except Exception as e:
                logger.error(f"背景重載發生錯誤：{e}")
            await asyncio.sleep(60)

    asyncio.create_task(_periodic_reload())


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """
    捕捉 Pydantic 驗證錯誤，回 HTTP 400 並回傳易懂訊息
    """
    return JSONResponse(
        status_code=400,
        content={"predictions": [], "error": "請求參數驗證失敗: " + str(exc)}
    )


@app.get("/")
@app.head("/")
async def root():
    """
    根路由：回傳簡單狀態，避免 404 被外部定期 HEAD 拋錯
    """
    return {"status": "OK"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    /predict Endpoint：
    1. 驗證請求內容（grid & target）
    2. 把 target 在 grid 中臨時遮蔽
    3. 收集各模組分數、正規化、融合（根據當前卡片尺寸選權重）
    4. 取得「歷史先驗」prior
    5. 加權融合 fused_scores + prior → final_scores
    6. 取 Top-3 並回傳
    """
    # --- 1) 驗證 grid 是否為非空矩形且尺寸 ≤ 50×50 ---
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

    # 驗證每個格位值：必須是 int，且要么 -1，要么 > 0，不可重複
    seen = set()
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
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

    # --- 2) 開始推論流程 ---
    try:
        arr = np.array(grid, dtype=int)

        # 2.1) 找到 target 所在位置 → 臨時遮蔽
        positions = np.argwhere(arr == target)
        if positions.shape[0] == 0:
            logger.error("Grid 中找不到 target")
            raise HTTPException(status_code=400, detail="Grid 中找不到 target")
        r0, c0 = int(positions[0][0]), int(positions[0][1])
        arr[r0, c0] = -1

        # 2.2) collect_all_scores + normalize
        tensor = collect_all_scores(arr, request_id="API")
        if tensor.size == 0:
            logger.error("No modules returned any scores")
            raise HTTPException(status_code=500, detail="No scoring modules available")
        tensor_norm = normalize_tensor(tensor, method="minmax")

        # 2.3) 根據 shape 取得加權向量 → fused_scores
        weights_dict = get_weights_for_shape((rows, cols))
        if weights_dict:
            module_names = list(REGISTERED_MODULES_BRAIN.keys())
            weights_list = [weights_dict.get(name, 0.0) for name in module_names]
            fused_scores = fuse_scores(tensor_norm, weights=weights_list)
        else:
            fused_scores = fuse_scores(tensor_norm, weights=None)

        # 2.4) 取得先驗 prior（histogram of true_pos for this shape,target）
        prior = get_target_prior((rows, cols), target)  # shape=(rows,cols)

        # 2.5) 加權融合 fused_scores 與 prior → final_scores
        α = 0.7
        final_scores = α * fused_scores + (1 - α) * prior

        # 2.6) 取 Top-3（只在 arr == -1 的空格挑分數最高）
        topk = get_topk_positions(final_scores, arr, k=3)
        predictions = []
        total = float(np.nansum(final_scores[arr == -1])) if np.any(arr == -1) else 1.0
        for (rr, cc), sc in topk:
            confidence = float(sc / total) if total > 0 else 0.0
            predictions.append({
                "row": rr + 1,   # 回傳 1-based index
                "col": cc + 1,
                "confidence": round(confidence, 6)
            })

        logger.info(f"[Predict] Top-3 => {predictions}")
        return {"predictions": predictions, "error": None}

    except HTTPException as e:
        # 已知錯誤直接拋出
        raise e
    except Exception as e:
        logger.exception("未知錯誤於 /predict")
        raise HTTPException(status_code=500, detail="Internal server error")
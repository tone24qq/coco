# main14.py
"""
main14.py：FastAPI 應用，提供 /predict Endpoint，調用 analyzer11 執行完整推論流程。
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
from analyzer11 import collect_all_scores, normalize_tensor, fuse_scores, get_topk_positions

# 設定日誌輸出
logging.basicConfig(
    filename="predict.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

class PredictRequest(BaseModel):
    grid: list[list[int]]
    target: int

class PredictResponse(BaseModel):
    predictions: list[dict]
    error: str | None = None

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    grid = req.grid
    target = req.target

    # 驗證 grid 是否為非空的矩形且尺寸 <= 50×50
    rows = len(grid)
    if rows == 0:
        logger.error("收到空 Grid")
        raise HTTPException(status_code=400, detail="Grid 不能為空")
    cols = len(grid[0])
    for row in grid:
        if len(row) != cols:
            logger.error("Grid 不是矩形")
            raise HTTPException(status_code=400, detail="Grid 必須為矩形")
    if rows > 50 or cols > 50:
        logger.error(f"Grid 大小 {rows}x{cols} 超過 50x50")
        raise HTTPException(status_code=400, detail="Grid 大小超過 50×50 限制")

    # 驗證格位值：必須是 int，且要么為 -1，要么 > 0
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

    # 執行推論
    try:
        arr = np.array(grid, dtype=int)
        blank_count = int(np.sum(arr == -1))
        logger.info(f"[Predict] target={target}, grid={rows}x{cols}, blanks={blank_count}")

        # 1) 蒐集所有模組分數 (tensor shape (44, rows, cols))
        tensor = collect_all_scores(arr, request_id="API")
        if tensor.size == 0:
            logger.error("No modules returned any scores")
            raise HTTPException(status_code=500, detail="No scoring modules available")

        # 2) 正規化
        tensor_norm = normalize_tensor(tensor, method="minmax")
        # 3) 融合 (此處預設等權平均，可自行調整權重)
        fused = fuse_scores(tensor_norm, weights=None)
        # 4) 取 Top-3 位置 (0-based)
        topk = get_topk_positions(fused, arr, k=3)
        predictions = []
        for (r, c), score in topk:
            predictions.append({
                "row": r + 1,      # 轉成 1-based
                "col": c + 1,
                "confidence": round(score, 6)
            })

        logger.info(f"[Predict] Top-3 => {predictions}")
        return {"predictions": predictions, "error": None}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("未知錯誤於 /predict")
        raise HTTPException(status_code=500, detail="Internal server error")
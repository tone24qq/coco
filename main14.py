# main14.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, validator, conint, Field
import numpy as np
import logging
from analyzer11 import collect_all_scores, normalize_tensor, fuse_scores, get_topk_positions

# ---------------- 日志 & 常量 ----------------
logging.basicConfig(
    filename="predict.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

MAX_ROWS = 50
MAX_COLS = 50

# ---------------- Pydantic 数据模型 ----------------
class AnalyzeRequest(BaseModel):
    # grid: 最外层是 1…MAX_ROWS 行，每行又是 1…MAX_COLS 列的整数列表
    grid: conlist(
        conlist(int, min_length=1, max_length=MAX_COLS),  # 每行 min_length=1, max_length=MAX_COLS
        min_length=1,
        max_length=MAX_ROWS                                # 总行数 min_length=1, max_length=MAX_ROWS
    )
    target: conint(gt=0)

    @validator("grid")
    def check_rectangular(cls, v):
        """
        确保 grid 是矩形：所有行的长度都相等
        """
        first_len = len(v[0])
        for row in v:
            if len(row) != first_len:
                raise ValueError("Grid 必须是矩形 (所有行长度相同)")
        return v

    @validator("target")
    def check_target_not_in_grid(cls, t, values):
        """
        确保 target 不在 grid 中已经存在的正整数里
        """
        grid = values.get("grid")
        if grid:
            flat = [val for row in grid for val in row if val != -1]
            if t in flat:
                raise ValueError("Target 已存在于 Grid 中")
        return t


class AnalyzeResponse(BaseModel):
    predictions: list[dict] = Field(..., description="Top‐3 建议位置列表")
    error: str | None = Field(None, description="如果发生错误则返回字符串，否则为 null")


# ---------------- FastAPI 应用初始化 ----------------
app = FastAPI(
    title="刮卡分析 API",
    version="1.0.0",
    description="基于向量化模块+历史先验，预测刮刮卡被遮空格最可能的数字 (Top‐3)"
)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    grid = req.grid
    target = req.target

    rows = len(grid)
    cols = len(grid[0])
    # 再次检查尺寸
    if rows > MAX_ROWS or cols > MAX_COLS:
        raise HTTPException(status_code=400, detail=f"Grid 大小超出 {MAX_ROWS}×{MAX_COLS} 限制")

    arr = np.array(grid, dtype=int)
    mask_missing = (arr == -1)
    blank_count = int(np.sum(mask_missing))
    logger.info(f"[Analyze] target={target}, grid={rows}x{cols}, blanks={blank_count}")

    if blank_count == 0:
        # 没有 -1，就无法预测
        return {"predictions": [], "error": None}

    try:
        # 1) 收集所有模块分数
        tensor = collect_all_scores(arr, request_id="API")
        if tensor.size == 0:
            return {"predictions": [], "error": "无可用评分模块"}

        # 2) 标准化
        tensor_norm = normalize_tensor(tensor, method="minmax")
        # 3) 融合
        fused = fuse_scores(tensor_norm, weights=None)
        # 4) Top‐3
        topk = get_topk_positions(fused, arr, k=3)

        preds = []
        for (r, c, conf) in topk:
            preds.append({
                "row": r + 1,
                "col": c + 1,
                "confidence": round(conf, 6)
            })
        return {"predictions": preds, "error": None}

    except Exception:
        logger.exception("分析失败")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@app.post("/", response_model=AnalyzeResponse)
async def root_analyze(req: AnalyzeRequest):
    # 根路径直接复用 analyze 逻辑
    return await analyze(req)
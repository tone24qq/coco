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

# ---------------- Pydantic Data Models ----------------
class AnalyzeRequest(BaseModel):
    # 先定义最外层的 grid：最多 50 行，每行也是一个 conlist
    grid: conlist(
        conlist(int, min_length=1, max_length=MAX_COLS),  # 每一行：最少 1 列，最多 MAX_COLS 列
        min_length=1,
        max_length=MAX_ROWS  # 最少 1 行，最多 MAX_ROWS 行
    )
    # target 必须 > 0
    target: conint(gt=0)

    @validator("grid")
    def check_rectangular(cls, v):
        """
        确认 grid 是矩形：所有行的长度都相同。
        这里 v 是一个 List[List[int]]
        """
        row0_len = len(v[0])
        for row in v:
            if len(row) != row0_len:
                raise ValueError("Grid 必须是矩形 (所有行长度相同)")
        return v

    @validator("target")
    def check_target_not_in_grid(cls, t, values):
        """
        确认 target 不在 grid 中的已知正整数里
        """
        grid = values.get("grid")
        if grid:
            flat_vals = [val for row in grid for val in row if val != -1]
            if t in flat_vals:
                raise ValueError("Target 已存在于 Grid 中")
        return t


class AnalyzeResponse(BaseModel):
    predictions: list[dict] = Field(..., description="Top‐3 建议位置列表")
    error: str | None = Field(None, description="如果发生错误则为字符串，否则为 null")


# ---------------- FastAPI App 初始化 ----------------
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
    # 再次检查尺寸 (虽然 Pydantic 已经保证行数、列数不超)
    if rows > MAX_ROWS or cols > MAX_COLS:
        raise HTTPException(status_code=400, detail=f"Grid 大小超出 {MAX_ROWS}×{MAX_COLS} 限制")

    # 转为 numpy array
    arr = np.array(grid, dtype=int)
    mask_missing = (arr == -1)
    blank_count = int(np.sum(mask_missing))
    logger.info(f"[Analyze] target={target}, grid={rows}x{cols}, blanks={blank_count}")

    # 如果没有任何 -1，就无法预测空格
    if blank_count == 0:
        return {"predictions": [], "error": None}

    try:
        # 1) 收集所有模块分数
        tensor = collect_all_scores(arr, request_id="API")
        if tensor.size == 0:
            return {"predictions": [], "error": "无可用评分模块"}

        # 2) 正规化
        tensor_norm = normalize_tensor(tensor, method="minmax")
        # 3) 融合 （默认等权）
        fused = fuse_scores(tensor_norm, weights=None)
        # 4) 获取 Top-3
        topk = get_topk_positions(fused, arr, k=3)

        # 组装返回结果
        preds = []
        for (r, c, conf) in topk:
            preds.append({
                "row": r + 1,      # 转为 1-based
                "col": c + 1,
                "confidence": round(conf, 6)
            })
        return {"predictions": preds, "error": None}

    except Exception as e:
        logger.exception("分析失败")
        raise HTTPException(status_code=500, detail="服务器内部错误")


# ------------ 新增根目录 POST / 路由（与 /analyze 逻辑相同） ------------
@app.post("/", response_model=AnalyzeResponse)
async def root_analyze(req: AnalyzeRequest):
    # 直接重复调用 analyze() 的逻辑
    return await analyze(req)
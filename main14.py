# main14.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, conint
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

# ---------------- Pydantic 模型 ----------------
class AnalyzeRequest(BaseModel):
    grid: conlist(conlist(int, min_items=1, max_items=MAX_COLS), min_items=1, max_items=MAX_ROWS)
    target: conint(gt=0)

    # 验证 grid 矩形
    @classmethod
    def validate_grid(cls, v):
        row_len = len(v[0])
        for row in v:
            if len(row) != row_len:
                raise ValueError("Grid 必须为矩形 (所有行长度相同)")
        return v

    # 验证 target 不在 grid 中
    @classmethod
    def validate_target(cls, v, values):
        grid = values.get("grid", [])
        flat = [val for row in grid for val in row if val != -1]
        if v in flat:
            raise ValueError("Target 已存在于 Grid 中")
        return v

    # Pydantic 目前版本使用 validator 装饰器写法：
    @staticmethod
    def __get_validators__():
        yield from BaseModel.__get_validators__()

    @classmethod
    def __modify_schema__(cls, field_schema):
        # 如果使用更高版本 pydantic，可用 @validator 装饰器替代
        pass


class AnalyzeResponse(BaseModel):
    predictions: list[dict]
    error: str | None = None


# ---------------- FastAPI App 初始化 ----------------
app = FastAPI(
    title="刮卡分析 API",
    version="1.0.0",
    description="基于向量化模块+历史先验，预测刮刮卡被遮空格最可能的数字 (Top‐3)"
)

# 原来的 /analyze 路由
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    grid = req.grid
    target = req.target

    rows = len(grid)
    cols = len(grid[0])
    # 验证 grid 尺寸
    if rows > MAX_ROWS or cols > MAX_COLS:
        raise HTTPException(status_code=400, detail=f"Grid 大小超出 {MAX_ROWS}×{MAX_COLS} 限制")

    # 转 numpy
    arr = np.array(grid, dtype=int)
    mask_missing = (arr == -1)
    blank_count = int(np.sum(mask_missing))
    logger.info(f"[Analyze] target={target}, grid={rows}x{cols}, blanks={blank_count}")

    if blank_count == 0:
        # 没有任何 -1→无法预测
        return {"predictions": [], "error": None}

    try:
        tensor = collect_all_scores(arr, request_id="API")
        if tensor.size == 0:
            return {"predictions": [], "error": "无可用评分模块"}
        tensor_norm = normalize_tensor(tensor, method="minmax")
        fused = fuse_scores(tensor_norm, weights=None)
        topk = get_topk_positions(fused, arr, k=3)
        preds = []
        for (r, c, conf) in topk:
            # 转成 1-based
            preds.append({"row": r + 1, "col": c + 1, "confidence": round(conf, 6)})
        return {"predictions": preds, "error": None}
    except Exception as e:
        logger.exception("分析失败")
        raise HTTPException(status_code=500, detail="服务器内部错误")


# **新增的根目录 POST / 路由**
# 它与 /analyze 一模一样，只是路径变成 "/"
@app.post("/", response_model=AnalyzeResponse)
async def root_analyze(req: AnalyzeRequest):
    # 直接调用 analyze() 里的逻辑
    return await analyze(req)
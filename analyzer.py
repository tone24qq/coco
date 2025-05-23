# analyzer.py

import os
import json
import time
import logging
from typing import List, Optional, Tuple, Callable, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -------------------------
# 配置和初始化
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyzer")

app = FastAPI(title="Grid Analyzer API")

# 默认从环境变量或本地文件加载记忆样本
MEMORY_FILE = os.getenv("MEMORY_CARDS_FILE", "memory_cards.json")
try:
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        _mem = json.load(f).get("memory_cards", [])
        MEMORY_CARDS = np.array(_mem, dtype=int)
        logger.info(f"Loaded memory cards: shape={MEMORY_CARDS.shape}")
except Exception as e:
    logger.warning(f"Cannot load memory cards ({MEMORY_FILE}): {e}")
    MEMORY_CARDS = np.empty((0,))

# -------------------------
# 数据模型
# -------------------------
class ProposedValue(BaseModel):
    pos: List[int] = Field(..., description="[row, col], 0-based")
    value: int = Field(..., ge=1, description="预测数字")
    score: float = Field(..., ge=0, description="置信分")
    py_guide: str = Field(..., description="本地推理说明")

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]] = Field(..., description="待补全的 NxM 矩阵，空格用 -1")
    targets: Optional[List[int]] = Field(
        None, description="要预测的行索引列表，默认所有含 -1 的行"
    )
    memory_override: Optional[List[List[List[int]]]] = Field(
        None, description="可选：临时覆盖记忆样本"
    )

class AnalyzeResponse(BaseModel):
    status: str
    visual_grid: str
    results: Dict[int, List[ProposedValue]]

# -------------------------
# 工具函数
# -------------------------
def to_numpy(grid: List[List[int]]) -> np.ndarray:
    arr = np.array(grid, dtype=int)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("new_card 必须为 2D 非空矩阵")
    return arr

def visualize_grid(arr: np.ndarray) -> str:
    rows, cols = arr.shape
    # 列标签
    header = "    " + " ".join(f"C{c+1:>3}" for c in range(cols)) + "\n"
    sep = "   +" + "+".join(["----"] * cols) + "+\n"
    body = ""
    for r in range(rows):
        row_vals = "".join(f"|{arr[r, c]:>3}" if arr[r, c] != -1 else "|   " for c in range(cols))
        body += f"R{r+1:>2} {row_vals}|\n" + sep
    return "```text\n" + header + sep + body + "```"

# -------------------------
# 模块化逻辑函数（示例）
# -------------------------
# 所有函数签名统一： (grid, pos, value) -> score_or_bool
def m1_fixed_empty(grid: np.ndarray, pos: Tuple[int,int], value: int) -> float:
    # 如果该格本来就空，打分 1，否则 0
    return 1.0 if grid[pos] == -1 else 0.0

def m2_interval_consistency(grid: np.ndarray, pos: Tuple[int,int], value: int) -> float:
    r, c = pos
    row = grid[r, :]
    # 找同值间距，越小越符合
    idx = np.where(row == value)[0]
    if idx.size == 0:
        return 0.0
    dist = np.min(np.abs(idx - c))
    return 1.0 / (1 + dist)

def m3_no_repeat(grid: np.ndarray, pos: Tuple[int,int], value: int) -> float:
    # 整卡不允许重复
    return 1.0 if not np.any(grid == value) else 0.0

# … 这里可继续添加更多模块 …
MODULES: List[Tuple[str, Callable[[np.ndarray, Tuple[int,int], int], float]]] = [
    ("fixed_empty", m1_fixed_empty),
    ("interval", m2_interval_consistency),
    ("no_repeat", m3_no_repeat),
    # ("adj_density", m4_adj_density), ...
]

# -------------------------
# 主分析逻辑
# -------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    t0 = time.time()
    # 转 numpy，校验
    try:
        grid = to_numpy(req.new_card)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 选择记忆样本
    memory = (
        np.array(req.memory_override, dtype=int)
        if req.memory_override is not None
        else MEMORY_CARDS
    )

    # 默认 targets：所有含 -1 的行
    if req.targets:
        targets = req.targets
    else:
        targets = list({r for r, row in enumerate(grid) if np.any(row == -1)})

    results: Dict[int, List[ProposedValue]] = {}

    # 对每个目标行，遍历所有可能候选值
    for r in targets:
        empties = list(zip(*np.where(grid[r, :] == -1)))
        for c in empties:
            candidates = set(range(1, grid.size + 1)) - set(grid.flatten())
            scored: List[Tuple[int,float,str]] = []
            for v in candidates:
                # 每个模块打分并求加权和（目前平均）
                scores = [fn(grid, (r,c), v) for _, fn in MODULES]
                avg_score = float(np.mean(scores))
                guide = ",".join(f"{name}:{s:.2f}" for (name,_), s in zip(MODULES, scores))
                scored.append((v, avg_score, guide))
            # 取 Top3
            top3 = sorted(scored, key=lambda x: -x[1])[:3]
            results.setdefault(r, []).extend([
                ProposedValue(pos=[r, c[1]], value=v, score=s, py_guide=g)
                for v, s, g in top3
            ])

    visual = visualize_grid(grid)
    elapsed = (time.time() - t0) * 1000
    logger.info(f"analyze done in {elapsed:.1f}ms")

    return AnalyzeResponse(
        status="success",
        visual_grid=visual,
        results=results
    )
# app.py
"""
REST API 入口
=============
提供：
    • /predict  :   解析單一網格，回傳 Top-3 推測
    • /batch    :   批次預測
    • /upload   :   上傳單一 JSON 或 ZIP (含 heatmap JSON) → 自動併入樣本索引
    • /health   :   Render / K8s probe

環境變數
--------
DATA_DIR        : 預設 "samples/data"
INDEX_DIR       : 預設 "samples/index"
MAX_HEATMAPS    : 限制索引樣本量 (預設 500_000)
USE_FAISS       : "1" 啟用 Faiss (若安裝)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

# 專案內模組
from analyzer import predict_scratch_card

# ────────────────────────────────────────────────────────────
# 環境與路徑
# ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "samples" / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "samples" / "index"))
MAX_HEATMAPS = int(os.getenv("MAX_HEATMAPS", "500000"))
USE_FAISS = os.getenv("USE_FAISS", "1") == "1"

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s:%(name)s] %(message)s",
    handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()],
)
logger = logging.getLogger("app")

# ────────────────────────────────────────────────────────────
# 嘗試載入 Faiss
# ────────────────────────────────────────────────────────────
try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FAISS_AVAILABLE = False
    USE_FAISS = False
    logger.warning("faiss 未安裝，降級為純 Python 索引。")

if USE_FAISS and not FAISS_AVAILABLE:
    logger.warning("USE_FAISS=1 但系統缺少 faiss，將改用純 Python。")
    USE_FAISS = False

FAISS_INDEX_PATH = INDEX_DIR / "faiss.idx"
META_PATH = INDEX_DIR / "meta_paths.json"

# ────────────────────────────────────────────────────────────
# Pydantic Schema
# ────────────────────────────────────────────────────────────
class GridInput(BaseModel):
    grid: List[List[int]] = Field(..., description="2D int list；-1 表空格")

    @model_validator(mode="after")
    def _check_matrix(self) -> "GridInput":
        if not self.grid or not self.grid[0]:
            raise ValueError("grid 為空")
        widths = {len(r) for r in self.grid}
        if len(widths) != 1:
            raise ValueError("grid 需為矩形，各列長度相同")
        return self


class BatchInput(BaseModel):
    items: List[GridInput]


class PredictResponse(BaseModel):
    iterations: int
    predictions: List[Dict[str, Any]]
    distribution: Dict[str, Dict[str, float]]


# ────────────────────────────────────────────────────────────
# FastAPI 初始化
# ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Scratch Card Analysis API",
    version="2.0.0",
    description="Lottery scratch-card hidden-number predictor",
    openapi_version="3.1.0",
)

# ────────────────────────────────────────────────────────────
# 索引載入與重建
# ────────────────────────────────────────────────────────────
heatmap_paths: List[str] = []            # All JSON paths
faiss_index: "faiss.Index" | None = None  # type: ignore
vector_dim = 0


def _extract_vector(path: Path) -> np.ndarray:
    """讀 JSON heatmap → 取 flatten 向量；長度不一致時補 0."""
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    vec = np.array(data, dtype=np.float32).ravel()
    global vector_dim
    if vector_dim == 0:
        vector_dim = vec.size
    if vec.size < vector_dim:
        vec = np.pad(vec, (0, vector_dim - vec.size))
    elif vec.size > vector_dim:
        vec = vec[:vector_dim]
    return vec


def build_faiss_index() -> None:
    """掃描所有 heatmap JSON → 建立 / 更新 Faiss 索引。"""
    global faiss_index, heatmap_paths, vector_dim

    logger.info("開始重建 Faiss 索引…")
    json_files = list(DATA_DIR.rglob("*.json"))
    heatmap_paths = [str(p) for p in json_files if "heatmap" in p.name]
    if not heatmap_paths:
        logger.warning("找不到 heatmap JSON，跳過索引。")
        return

    if FAISS_AVAILABLE:
        sample_vec = _extract_vector(Path(heatmap_paths[0]))
        vector_dim = sample_vec.size
        faiss_index = faiss.IndexFlatL2(vector_dim)
        metas: List[str] = []

        for i, p in enumerate(heatmap_paths):
            if i >= MAX_HEATMAPS:
                break
            vec = _extract_vector(Path(p))
            faiss_index.add(vec.reshape(1, -1))
            metas.append(p)

            if i % 10000 == 0 and i:
                logger.info("已索引 %d 筆…", i)

        faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
        META_PATH.write_text(json.dumps(metas, ensure_ascii=False))
        logger.info("Faiss 索引完成，向量 %d", len(metas))
    else:
        logger.info("純 Python 模式：紀錄 heatmap 路徑 %d 筆", len(heatmap_paths))


@app.on_event("startup")
async def _startup() -> None:
    logger.info("應用啟動，執行初始化任務")
    await asyncio.to_thread(build_faiss_index)


# ────────────────────────────────────────────────────────────
# 內部工具
# ────────────────────────────────────────────────────────────
def _save_uploaded_file(file: UploadFile, dest: Path) -> Path:
    with dest.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)
    return dest


def _extract_zip(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(zip_path.with_suffix(""))


# ────────────────────────────────────────────────────────────
# API Routes
# ────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: GridInput) -> JSONResponse:
    result = await asyncio.to_thread(predict_scratch_card, req.grid)
    return JSONResponse(content=result)


@app.post("/batch", response_model=List[PredictResponse])
async def batch_endpoint(batch: BatchInput) -> JSONResponse:
    results = await asyncio.gather(
        *[asyncio.to_thread(predict_scratch_card, item.grid) for item in batch.items]
    )
    return JSONResponse(content=results)


@app.post("/upload")
async def upload_endpoint(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> JSONResponse:
    """
    上傳 .json 或 .zip：
        • .json               → 直接移到 DATA_DIR
        • .zip (含 heatmap)   → 解壓至 DATA_DIR/zip_name
    上傳完成後背景重建索引。
    """
    if file.content_type not in ("application/zip", "application/json", "text/json"):
        raise HTTPException(status_code=415, detail="僅接受 .json 或 .zip")

    dest_path = DATA_DIR / file.filename
    _save_uploaded_file(file, dest_path)

    if dest_path.suffix.lower() == ".zip":
        _extract_zip(dest_path)

    background_tasks.add_task(build_faiss_index)
    return JSONResponse({"status": "ok", "filename": file.filename})


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ────────────────────────────────────────────────────────────
# 本地啟動
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
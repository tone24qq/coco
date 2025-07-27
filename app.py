import glob
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # torch may be unavailable in minimal runtimes
    torch = None  # type: ignore[assignment]
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from dataset import BLANK_VALUE
from model import DynamicMET
from utils import ensure_only_blank

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Matrix Factorization Service", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "coco", "version": "0.1.0"}


@app.get("/health")
def health():
    """Simple readiness/liveness probe.

    Returns the loaded model shapes so ops can verify startup state.
    """
    return {
        "status": "ok",
        "models": [{"rows": r, "cols": c} for (r, c) in models.keys()],
    }


class PredictRequest(BaseModel):
    board: List[List[int]] = Field(
        ..., description=f"2D grid, blanks use {BLANK_VALUE}."
    )
    # 兩個欄位都接受，擇一或兩者皆送都行
    target: Optional[int] = Field(None, description="Preferred. Target number.")
    target_value: Optional[int] = Field(
        None,
        description="Alias of `target`. Accepted for backward compatibility.",
    )

    @model_validator(mode="after")
    def _normalize(cls, values: "PredictRequest"):
        if values.target is None and values.target_value is None:
            raise ValueError("One of `target` or `target_value` must be provided.")
        if values.target is None and values.target_value is not None:
            values.target = values.target_value
            logger.warning("[REQ] using `target_value` as `target`: %s", values.target)
        elif values.target is not None and values.target_value is not None:
            if values.target != values.target_value:
                raise ValueError(
                    f"Inconsistent target: target={values.target} != target_value={values.target_value}"
                )
            logger.info(
                "[REQ] both `target` and `target_value` provided, value=%s",
                values.target,
            )
        return values


class Prediction(BaseModel):
    row: int
    col: int
    score: float  # 0~1 之間的置信度


MODEL_GLOB = os.environ.get("MODEL_GLOB", os.path.join("checkpoints", "met_*x*.pth"))
_PATTERN = re.compile(r"met_(\d+)x(\d+)\.pth$")
models: Dict[Tuple[int, int], DynamicMET] = {}


def _load_one(path: str, rows: int, cols: int) -> DynamicMET:
    n = rows * cols
    model = DynamicMET(n, n)
    if torch is not None and os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        if hasattr(model, "eval"):
            model.eval()
        logger.info(
            "載入模型檔案 %s，尺寸 %sx%s", path, rows, cols
        )  # 中文log：載入已存在的模型
    else:
        logger.info(
            "建立新模型，尺寸 %sx%s", rows, cols
        )  # 中文log：未找到檔案時新建模型
    return model


def _discover_models() -> None:
    """Locate all checkpoint files under ``MODEL_GLOB`` and load them."""
    found = False
    logger.info("開始搜尋模型檔案，樣式: %s", MODEL_GLOB)  # 中文log：啟動時掃描模型檔
    for path in glob.glob(MODEL_GLOB):
        m = _PATTERN.match(os.path.basename(path))
        if not m:
            continue
        r, c = int(m.group(1)), int(m.group(2))
        models[(r, c)] = _load_one(path, r, c)
        found = True
    if not found:
        r, c = 8, 10
        models[(r, c)] = DynamicMET(r * c, r * c)
        if hasattr(models[(r, c)], "eval"):
            models[(r, c)].eval()
        logger.warning(
            "未找到模型檔案，使用預設模型 %sx%s", r, c
        )  # 中文log：沒有模型時使用預設


@app.on_event("startup")
def load_models() -> None:
    """Search and load checkpoints following pattern met_{R}x{C}.pth."""
    logger.info("應用啟動，準備載入模型")  # 中文log：啟動事件
    _discover_models()


@app.post("/predict", response_model=List[Prediction])
def predict(req: PredictRequest):
    if not req.board or not all(isinstance(r, list) and r for r in req.board):
        raise HTTPException(
            status_code=422, detail="`board` must be a non-empty 2D list."
        )
    lens = {len(r) for r in req.board}
    if len(lens) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"`board` rows must have equal length, got lengths={sorted(lens)}",
        )

    board = np.asarray(req.board, dtype=int)
    uniq, cnt = np.unique(board, return_counts=True)
    logger.info(
        "[CHK] uniq=%s cnt=%s BLANK_VALUE=%s", uniq.tolist(), cnt.tolist(), BLANK_VALUE
    )
    rows, cols = board.shape
    n = rows * cols
    flat_all = board.flatten()
    valid_values = set(range(1, n + 1))
    for v in flat_all:
        if v != BLANK_VALUE and v not in valid_values:
            raise HTTPException(status_code=422, detail="board values out of range")
    non_blank = flat_all[flat_all != BLANK_VALUE]
    if non_blank.size != len(set(non_blank.tolist())):
        raise HTTPException(status_code=422, detail="board has duplicate numbers")
    target = req.target
    if target is None:
        raise HTTPException(status_code=422, detail="`target` is required.")
    if not (1 <= target <= n):
        raise HTTPException(
            status_code=422, detail=f"target must be in [1, {n}], got {target}"
        )
    logger.info(
        "盤面尺寸 %sx%s，目標數字 %s", rows, cols, target
    )  # 中文log：記錄盤面大小與目標數字

    flat = board.flatten()
    mask_pos = np.where(flat == BLANK_VALUE)[0]
    if mask_pos.size == 0:
        raise HTTPException(
            status_code=422, detail=f"no blank cells ({BLANK_VALUE}) to predict"
        )
    logger.info("[CHK] mask_pos=%s", mask_pos.tolist())

    # ensure contiguous int64 array to avoid torch dtype inference errors
    flat_input = np.where(flat < 0, 0, flat).astype(np.int64, copy=False)
    flat_input = np.ascontiguousarray(flat_input)
    logger.info(
        "[CHK] flat_input: shape=%s dtype=%s sample=%s",
        flat_input.shape,
        flat_input.dtype,
        flat_input[: min(10, flat_input.size)].tolist(),
    )
    # 中文log：檢查輸入資料

    model = models.get((rows, cols))
    if model is None:
        logger.info(
            "尚未載入 %sx%s 的模型，立即建立", rows, cols
        )  # 中文log：動態建立模型
        model = DynamicMET(n, n)
        if hasattr(model, "eval"):
            model.eval()
        models[(rows, cols)] = model
    else:
        logger.info("使用已載入的模型 %sx%s", rows, cols)  # 中文log：重複尺寸共用模型

    if torch is not None:
        # use as_tensor to avoid copy and specify dtype explicitly
        inp = torch.as_tensor(flat_input, dtype=torch.long).unsqueeze(0)
        logits = model(inp)  # type: ignore[misc]
        probs = torch.softmax(logits, dim=-1)
        logger.info(
            "[PRED] torch path: inp=%s logits=%s probs=%s",
            tuple(inp.shape),
            tuple(logits.shape),
            tuple(probs.shape),
        )
        V = probs.shape[-1]
        target_idx = target if V == n + 1 else target - 1
        if not (0 <= target_idx < V):
            raise HTTPException(
                status_code=422,
                detail=f"target index out of range: target={target}, mapped={target_idx}, V={V}",
            )
        scores_all = probs[0, :, target_idx]
        scores_np = scores_all.detach().cpu().numpy()
    else:
        inp = flat_input.reshape(1, -1)
        logits = model(inp)
        arr = np.asarray(logits)
        logger.info(
            "[PRED] numpy path: inp=%s logits=%s",
            inp.shape,
            arr.shape,
        )
        V = arr.shape[-1]
        target_idx = target if V == n + 1 else target - 1
        if not (0 <= target_idx < V):
            raise HTTPException(
                status_code=422,
                detail=f"target index out of range: target={target}, mapped={target_idx}, V={V}",
            )
        scores_np = arr[0, :, target_idx]

    candidate_scores = scores_np[mask_pos]
    topk_local = np.argsort(candidate_scores)[-min(3, len(candidate_scores)) :][::-1]
    logger.info(
        "從 %s 個候選格中挑選前 %s 名", len(candidate_scores), len(topk_local)
    )  # 中文log：候選格與返回數量
    top_indices = mask_pos[topk_local]
    logger.info("[CHK] top_indices=%s", top_indices.tolist())

    raw = [
        Prediction(
            row=int(idx // cols), col=int(idx % cols), score=float(scores_np[idx])
        )
        for idx in top_indices
    ]
    return ensure_only_blank(board, raw, BLANK_VALUE)

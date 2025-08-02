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

from dataset import BLANK_VALUE, MASK_TOKEN_ID
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
    # 方便前端與除錯
    idx: Optional[int] = None
    cell_value: Optional[int] = None


MODEL_GLOB = os.environ.get("MODEL_GLOB", os.path.join("checkpoints", "met_*x*.pth"))
_PATTERN = re.compile(r"met_(\d+)x(\d+)\.pth$")

# Training-time hyperparameters required for loading checkpoints
MODEL_PARAMS = {
    "d_model": 256,
    "nhead": 8,
    "depth": 8,
    "dropout": 0.1,
}

models: Dict[Tuple[int, int], DynamicMET] = {}


def _create_model(rows: int, cols: int) -> DynamicMET:
    """Return a :class:`DynamicMET` with training hyperparameters."""
    num_fields = rows * cols
    logger.info(
        "Model: shape=%sx%s, d_model=%s, depth=%s, nhead=%s, num_values=%s",
        rows,
        cols,
        MODEL_PARAMS["d_model"],
        MODEL_PARAMS["depth"],
        MODEL_PARAMS["nhead"],
        num_fields,
    )
    logger.info(
        "Mapping: BLANK_VALUE=%s -> MASK_TOKEN_ID=%s ; labels 1..%s",
        BLANK_VALUE,
        MASK_TOKEN_ID,
        num_fields,
    )
    det = False
    if torch is not None and hasattr(torch, "are_deterministic_algorithms_enabled"):
        det = torch.are_deterministic_algorithms_enabled()
    logger.info("Deterministic: %s", det)
    return DynamicMET(
        num_fields=num_fields,
        num_values=num_fields,
        rows=rows,
        cols=cols,
        **MODEL_PARAMS,
    )


def _load_one(path: str, rows: int, cols: int) -> DynamicMET:
    """Load checkpoint at ``path`` or initialize a new model."""

    model = _create_model(rows, cols)
    if torch is not None and os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        if hasattr(model, "eval"):
            model.eval()
        assert (
            model.classifier.out_features == model.num_values
        ), "classifier dim mismatch"
        logger.info(
            "載入模型檔案 %s，尺寸 %sx%s", path, rows, cols
        )  # 中文log：載入已存在的模型
    else:
        logger.info(
            "建立新模型，尺寸 %sx%s", rows, cols
        )  # 中文log：未找到檔案時新建模型
        assert (
            model.classifier.out_features == model.num_values
        ), "classifier dim mismatch"
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
        models[(r, c)] = _create_model(r, c)
        if hasattr(models[(r, c)], "eval"):
            models[(r, c)].eval()
        logger.warning(
            "未找到模型檔案，使用預設模型 %sx%s", r, c
        )  # 中文log：沒有模型時使用預設


@app.on_event("startup")
def load_models() -> None:
    """Search and load checkpoints following pattern met_{R}x{C}.pth."""
    root_logger = logging.getLogger()
    env_level = os.environ.get("LOG_LEVEL")
    if env_level:
        root_logger.setLevel(env_level.upper())
    else:
        root_logger.setLevel(logging.getLogger("uvicorn").level)
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
    flat_input = np.where(flat == BLANK_VALUE, MASK_TOKEN_ID, flat).astype(
        np.int64, copy=False
    )
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
        model = _create_model(rows, cols)
        if hasattr(model, "eval"):
            model.eval()
        models[(rows, cols)] = model
    else:
        logger.info("使用已載入的模型 %sx%s", rows, cols)  # 中文log：重複尺寸共用模型

    if torch is not None:
        # use as_tensor to avoid copy and specify dtype explicitly
        inp = torch.as_tensor(flat_input, dtype=torch.long).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            logits = model(inp)  # type: ignore[misc]
        logger.info(
            "IDX semantics: 0=blank(ignored), 1..%s=numbers 1..%s ; logits.shape=%s",
            n,
            n,
            tuple(logits.shape),
        )
        if logger.isEnabledFor(logging.DEBUG):
            n_f = logits.size(1)
            pos_ids = torch.arange(n_f, device=logits.device)
            row_ids = torch.div(pos_ids, cols, rounding_mode="floor")
            col_ids = pos_ids % cols
            logger.debug(
                "[RoPE] row_ids=%s col_ids=%s sample_q=%s",
                row_ids[:10].tolist(),
                col_ids[:10].tolist(),
                logits[0, :5, :5].detach().cpu().numpy().round(4),
            )
        probs = torch.softmax(logits, dim=-1)
        logger.info(
            "[PRED] torch path: inp=%s logits=%s probs=%s",
            tuple(inp.shape),
            tuple(logits.shape),
            tuple(probs.shape),
        )
        V = probs.shape[-1]
        target_idx = target
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
        logger.info(
            "IDX semantics: 0=blank(ignored), 1..%s=numbers 1..%s ; logits.shape=%s",
            n,
            n,
            arr.shape,
        )
        V = arr.shape[-1]
        target_idx = target
        if not (0 <= target_idx < V):
            raise HTTPException(
                status_code=422,
                detail=f"target index out of range: target={target}, mapped={target_idx}, V={V}",
            )
        scores_np = arr[0, :, target_idx]

    candidate_idx = mask_pos
    candidate_scores = scores_np[candidate_idx]
    k = min(3, len(candidate_scores))
    topk_local = np.argpartition(candidate_scores, -k)[-k:]
    order = np.lexsort((candidate_idx[topk_local], -candidate_scores[topk_local]))
    top_indices = candidate_idx[topk_local][order][-k:][::-1]
    logger.info("TopK: candidates = %s, k=%s", len(candidate_scores), k)
    logger.info("[CHK] top_indices=%s", top_indices.tolist())

    picked_vals = [int(flat[idx]) for idx in top_indices]
    # 中文 log：以 row-col 形式列出 top3 名次，並確認格子皆為空白
    pos_str = " ".join(
        f"{r}-{c}"
        for idx in top_indices
        for r, c in [np.unravel_index(int(idx), board.shape)]
    )
    logger.info("top3=%s %s格皆為空格（符合預期）", pos_str, len(top_indices))
    logger.info(
        "[CHK] picked vals=%s (should all be BLANK_VALUE=%s)",
        picked_vals,
        BLANK_VALUE,
    )
    violations = [
        (*np.unravel_index(int(idx), board.shape), int(flat[idx]))
        for idx in top_indices
        if flat[idx] != BLANK_VALUE
    ]
    if violations:
        logger.error("[FATAL] non-blank selected! violations=%s", violations)
        raise HTTPException(
            status_code=500,
            detail={"error": "non-blank-selected", "violations": violations},
        )

    raw = []
    for idx in top_indices:
        r, c = np.unravel_index(int(idx), board.shape)
        raw.append(
            Prediction(
                row=r,
                col=c,
                score=float(scores_np[idx]),
                idx=int(idx),
                cell_value=int(flat[idx]),
            )
        )
    validated = ensure_only_blank(board, raw, BLANK_VALUE)
    for item in validated:
        item.row += 1
        item.col += 1
    # 中文 log：列印前三名位置與機率百分比
    percent_msg = " ".join(
        f"top{i + 1}={item.row}-{item.col}({item.score * 100:.0f}%)"
        for i, item in enumerate(validated)
    )
    logger.info("預測機率 %s", percent_msg)
    return validated

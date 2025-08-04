import json
import os
import random
import threading
from pathlib import Path

import numpy as np

# 1. OS / Python level determinism
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(42)
np.random.seed(42)

try:
    import torch
except Exception:  # torch may be unavailable in minimal runtimes
    torch = None  # type: ignore[assignment]
else:  # pragma: no branch - executed when torch is available
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:  # pragma: no cover - older torch
        pass

import glob  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field, model_validator  # noqa: E402

from agents.memory_agent import build_memory as build_memory_agent  # noqa: E402
from agents.memory_agent import predict as memory_predict  # noqa: E402
from agents.memory_agent import predict_stream as memory_predict_stream  # noqa: E402
from dataset import BLANK_VALUE, MASK_TOKEN_ID, validate_board  # noqa: E402
from model import DynamicMET  # noqa: E402
from utils import ensure_only_blank, ensure_unique  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Matrix Factorization Service", version="0.1.0")
@app.post("/predict")
async def predict(req: PredictRequest):
    board = np.array(req.board, dtype=int)
    if np.all(board == BLANK_VALUE):
        # 👇 健康檢查專用：100 毫秒內回 200
        return {"status": "ok", "note": "all-blank health-check"}

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
    score: float  # 0~100 信心百分比
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
memories: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
memory_files: Dict[Tuple[int, int], Path] = {}

_mem_limit_env = os.environ.get("MEMORY_SAMPLE_LIMIT")
MEMORY_SAMPLE_LIMIT: Optional[int]
if _mem_limit_env:
    try:
        MEMORY_SAMPLE_LIMIT = int(_mem_limit_env)
    except ValueError:  # pragma: no cover - invalid env input
        MEMORY_SAMPLE_LIMIT = None
else:
    MEMORY_SAMPLE_LIMIT = None


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


def _load_memory_for_shape(rows: int, cols: int, model: DynamicMET) -> None:
    """Register memory source for ``(rows, cols)`` if archive exists."""
    jsonl_path = Path("data_archives") / f"{rows}x{cols}.jsonl"
    if jsonl_path.is_file():
        memory_files[(rows, cols)] = jsonl_path
        logger.info("記憶庫採流式讀取：%s", jsonl_path)
        return

    file_path = Path("data_archives") / f"{rows}x{cols}.json"
    if not file_path.is_file():
        logger.warning("未找到記憶庫檔案：%s", file_path)
        return
    data = json.load(open(file_path, "r", encoding="utf-8"))
    if MEMORY_SAMPLE_LIMIT is not None and len(data) > MEMORY_SAMPLE_LIMIT:
        logger.info("限制記憶庫樣本 %s -> %s", len(data), MEMORY_SAMPLE_LIMIT)
        data = data[:MEMORY_SAMPLE_LIMIT]
    samples = [(np.array(e["board"], dtype=int), int(e["target"])) for e in data]
    keys, values = build_memory_agent(samples, model)
    memories[(rows, cols)] = (keys, values)
    logger.info("✅ 記憶庫就緒：keys=%s values=%s", keys.shape, values.shape)


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


def _preload_memories() -> None:
    """Load memory banks in background after startup."""
    for (r, c), model in list(models.items()):
        try:
            _load_memory_for_shape(r, c, model)
        except Exception:
            logger.exception("記憶庫載入失敗：%sx%s", r, c)


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
    threading.Thread(target=_preload_memories, daemon=True).start()


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
    try:
        validate_board(board, allow_blank=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rows, cols = board.shape
    n = rows * cols
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

    model = models.get((rows, cols))
    if model is None:
        logger.info(
            "尚未載入 %sx%s 的模型，立即建立", rows, cols
        )  # 中文log：動態建立模型
        model = _create_model(rows, cols)
        if hasattr(model, "eval"):
            model.eval()
        models[(rows, cols)] = model
        _load_memory_for_shape(rows, cols, model)
    else:
        logger.info("使用已載入的模型 %sx%s", rows, cols)  # 中文log：重複尺寸共用模型

    memory = memories.get((rows, cols))
    jsonl = memory_files.get((rows, cols))
    if memory is None and jsonl is None:
        _load_memory_for_shape(rows, cols, model)
        memory = memories.get((rows, cols))
        jsonl = memory_files.get((rows, cols))
    if jsonl is not None:
        fused = memory_predict_stream(
            board.copy(),
            target=target,
            model=model,
            jsonl_path=jsonl,
            alpha=0.5,
            k_neighbors=2,
            topk=3,
        )
    elif memory is not None:
        memory_keys, memory_values = memory
        fused = memory_predict(
            board.copy(),
            target=target,
            model=model,
            memory_keys=memory_keys,
            memory_values=memory_values,
            alpha=0.5,
            k_neighbors=2,
            topk=3,
        )
    else:
        fused = []

    if fused:
        coords = []
        for item in fused:
            r0 = int(item["row"]) - 1
            c0 = int(item["col"]) - 1
            idx = r0 * cols + c0
            coords.append((r0, c0, float(item["score"]), idx))
        if coords:
            total = float(sum(sc for _, _, sc, _ in coords))
            if total > 0:
                norm_scores = [sc / total for _, _, sc, _ in coords]
            else:
                norm_scores = [1 / len(coords)] * len(coords)
            raw_preds = [
                Prediction(
                    row=r0,
                    col=c0,
                    score=round(ns * 100, 2),
                    idx=idx,
                    cell_value=BLANK_VALUE,
                )
                for (r0, c0, _, idx), ns in zip(coords, norm_scores)
            ]
            validated = ensure_only_blank(board, raw_preds, BLANK_VALUE)
            validated = ensure_unique(validated)
            expected = min(3, mask_pos.size)
            if len(validated) == expected:
                for item in validated:
                    item.row += 1
                    item.col += 1
                percent_msg = " ".join(
                    f"top{i + 1}={item.row}-{item.col}({item.score:.0f}%)"
                    for i, item in enumerate(validated)
                )
                logger.info("預測機率 %s", percent_msg)
                return validated

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
    assert (
        candidate_scores.shape[0] == candidate_idx.shape[0]
    ), f"空白格 {candidate_idx.shape[0]} 个，却只打分了 {candidate_scores.shape[0]} 个！"
    logger.debug("✅ 完整打分：空白格共 %s 个，已收到分数", candidate_idx.shape[0])

    coord_scores = []
    for idx, sc in zip(candidate_idx, candidate_scores):
        r, c = np.unravel_index(int(idx), board.shape)
        coord_scores.append((r, c, float(sc), int(idx)))
    coord_scores.sort(key=lambda x: (-x[2], x[0], x[1]))

    total = float(sum(item[2] for item in coord_scores))
    if total > 0:
        norm_scores = [item[2] / total for item in coord_scores]
    else:
        norm_scores = [1 / len(coord_scores)] * len(coord_scores)

    k = min(3, len(coord_scores))
    top_items = coord_scores[:k]
    logger.info("TopK: candidates = %s, k=%s", len(coord_scores), k)
    logger.info("[CHK] top_indices=%s", [it[3] for it in top_items])

    picked_vals = [int(flat[item[3]]) for item in top_items]
    # 中文 log：以 row-col 形式列出 top3 名次，並確認格子皆為空白
    pos_str = " ".join(f"{r}-{c}" for r, c, _, _ in top_items)
    logger.info("top3=%s %s格皆為空格（符合預期）", pos_str, len(top_items))
    logger.info(
        "[CHK] picked vals=%s (should all be BLANK_VALUE=%s)",
        picked_vals,
        BLANK_VALUE,
    )
    violations = [
        (r, c, int(flat[idx])) for r, c, _, idx in top_items if flat[idx] != BLANK_VALUE
    ]
    if violations:
        logger.error("[FATAL] non-blank selected! violations=%s", violations)
        raise HTTPException(
            status_code=500,
            detail={"error": "non-blank-selected", "violations": violations},
        )

    raw: List[Prediction] = []
    for i, (r, c, _, idx) in enumerate(top_items):
        raw.append(
            Prediction(
                row=r,
                col=c,
                score=round(float(norm_scores[i] * 100), 2),
                idx=idx,
                cell_value=int(flat[idx]),
            )
        )
    validated = ensure_only_blank(board, raw, BLANK_VALUE)
    validated = ensure_unique(validated)
    for item in validated:
        item.row += 1
        item.col += 1
    # 中文 log：列印前三名位置與機率百分比
    percent_msg = " ".join(
        f"top{i + 1}={item.row}-{item.col}({item.score:.0f}%)"
        for i, item in enumerate(validated)
    )
    logger.info("預測機率 %s", percent_msg)
    return validated

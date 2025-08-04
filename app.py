"""FastAPI application for matrix factorization service."""

# isort: skip_file

import os
import random
import time
from pathlib import Path

import numpy as np

# 1. OS / Python level determinism
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(42)
np.random.seed(42)

# limit streaming fallback to protect against huge archives
os.environ.setdefault("MEMORY_MAX_SCAN", "1000")

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
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field, conlist, model_validator  # noqa: E402

from agents.memory_agent import build_memory as build_memory_agent  # noqa: E402
from dataset import BLANK_VALUE, MASK_TOKEN_ID, validate_board  # noqa: E402
from model import DynamicMET  # noqa: E402
from utils import ensure_only_blank, ensure_unique  # noqa: E402

try:  # optional approximate nearest neighbor library
    import hnswlib  # noqa: E402
except Exception:  # pragma: no cover - allow environments without hnswlib
    hnswlib = None  # type: ignore[assignment]

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
    """Schema for prediction requests."""

    board: conlist(conlist(int, min_length=1), min_length=1) = Field(
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
memory_targets: Dict[Tuple[int, int], np.ndarray] = {}
hnsw_indices: Dict[Tuple[int, int], Any] = {}


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


def _load_memory_for_shape(rows: int, cols: int) -> None:
    """從 ``data_archives`` 載入 ``(rows, cols)`` 的記憶庫快取。"""

    base = Path("data_archives")
    paths = sorted(base.glob(f"{rows}x{cols}_memory*.npz"))
    if not paths:
        logger.warning("未找到記憶庫檔案：%s", base / f"{rows}x{cols}_memory*.npz")
        return

    keys_list: List[np.ndarray] = []
    values_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    for path in paths:
        data = np.load(path, mmap_mode="r")
        keys_list.append(data["keys"])
        values_list.append(data["values"])
        if "targets" in data.files:
            targets_list.append(data["targets"])

    keys = np.concatenate(keys_list, axis=0)
    values = np.concatenate(values_list, axis=0)
    memories[(rows, cols)] = (keys, values)
    if targets_list:
        memory_targets[(rows, cols)] = np.concatenate(targets_list, axis=0)
    if hnswlib is not None:
        idx = hnswlib.Index(space="l2", dim=keys.shape[1])
        idx.init_index(max_elements=len(keys), M=16, ef_construction=200)
        idx.add_items(keys, np.arange(len(keys)))
        hnsw_indices[(rows, cols)] = idx
    logger.info(
        "✅ 記憶庫載入：%s parts=%s keys=%s values=%s",
        f"{rows}x{cols}",
        len(paths),
        keys.shape,
        values.shape,
    )


def find_similar(
    rows: int,
    cols: int,
    board: np.ndarray,
    target: int,
    k: int = 3,
) -> List[Dict[str, Any]]:
    """Return top-``k`` similar samples.

    若已載入 HNSW 索引則優先使用；否則退回以 NumPy 計算所有距離。
    可搭配 :func:`filter_by_target` 先篩選目標，再對索引結果做距離排序。
    """

    model = models[(rows, cols)]
    q, _ = build_memory_agent([(board, target)], model)
    keys, _ = memories[(rows, cols)]
    idx = hnsw_indices.get((rows, cols))
    if idx is not None:
        labels, dists = idx.knn_query(q, k=k)
        labels, dists = labels[0], dists[0]
    else:
        vec = q[0]
        dists = np.linalg.norm(keys - vec, axis=1)
        labels = np.argsort(dists)[:k]
        dists = dists[labels]
    targets = memory_targets.get((rows, cols))
    sims: List[Dict[str, Any]] = []
    for lbl, dist in zip(labels, dists):
        item: Dict[str, float] = {"sample_idx": int(lbl), "distance": float(dist)}
        if targets is not None:
            item["target"] = int(targets[lbl])
        sims.append(item)
    return sims


def filter_by_target(rows: int, cols: int, target: int) -> List[int]:
    """Return indices of samples whose original target equals ``target``.

    可先呼叫本函式取得所有符合目標的樣本，再依距離排序以找出與
    目前盤面最相似的範例。
    """

    targets = memory_targets.get((rows, cols))
    if targets is None:
        raise KeyError(f"targets for {rows}x{cols} not loaded")
    return np.where(targets == target)[0].tolist()


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


def preload_memories() -> None:
    """在啟動時預先載入所有 ``*_memory*.npz``。"""

    loaded: List[str] = []
    base = Path("data_archives")
    seen: set[Tuple[int, int]] = set()
    for npz in base.glob("*x*_memory*.npz"):
        shape = npz.stem.split("_memory")[0]
        rows, cols = map(int, shape.split("x"))
        if (rows, cols) in seen:
            continue
        seen.add((rows, cols))
        if (rows, cols) not in models:
            model = _create_model(rows, cols)
            if hasattr(model, "eval"):
                model.eval()
            models[(rows, cols)] = model
        try:
            _load_memory_for_shape(rows, cols)
            if (rows, cols) in memories:
                loaded.append(f"{rows}x{cols}")
        except Exception:
            logger.exception("記憶庫載入失敗：%sx%s", rows, cols)
    if loaded:
        logger.info("共快取了以下 (rows×cols)：%s", ", ".join(loaded))


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
    preload_memories()


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
    else:
        logger.info("使用已載入的模型 %sx%s", rows, cols)  # 中文log：重複尺寸共用模型

    memory = memories.get((rows, cols))
    if memory is None:
        _load_memory_for_shape(rows, cols)
        memory = memories.get((rows, cols))

    mem_n = int(memory[0].shape[0]) if memory is not None else 0
    logger.info("記憶庫：尺寸=%sx%s 樣本=%d", rows, cols, mem_n)

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

    start_model = time.perf_counter()
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
    model_time = time.perf_counter() - start_model
    blank_cnt = int(mask_pos.size)
    logger.info("模型推理：空白格=%d 耗時=%.3f秒", blank_cnt, model_time)

    # step 2-3: memory fusion
    start_mem = time.perf_counter()
    final_scores = scores_np
    try:
        sim_indices = filter_by_target(rows, cols, target)
    except KeyError:
        sim_indices = []
    retrieved = len(sim_indices)
    if sim_indices and memory is not None:
        keys, values = memory
        q, _ = build_memory_agent([(board, target)], model)
        dists = np.linalg.norm(keys[sim_indices] - q[0], axis=1)
        k_mem = min(3, len(sim_indices))
        topk_idx = np.argsort(dists)[:k_mem]
        knn_scores = values[sim_indices][topk_idx]
        memory_score = np.mean(knn_scores, axis=0)
        alpha = 0.5
        final_scores = scores_np * alpha + memory_score * (1 - alpha)
        mem_time = time.perf_counter() - start_mem
        logger.info(
            "記憶檢索：樣本=%d 命中=%d 耗時=%.3f秒",
            mem_n,
            retrieved,
            mem_time,
        )
        logger.info(
            "合併：模型權重=%.1f 記憶權重=%.1f",
            alpha,
            1 - alpha,
        )
    else:
        mem_time = time.perf_counter() - start_mem
        logger.info("記憶檢索：樣本=%d 命中=0 耗時=%.3f秒", mem_n, mem_time)
        final_scores = scores_np
        logger.info("合併：僅模型輸出（無記憶）")

    candidate_idx = mask_pos
    candidate_scores = final_scores[candidate_idx]
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
    logger.info("前K名：候選=%s，k=%s", len(coord_scores), k)
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

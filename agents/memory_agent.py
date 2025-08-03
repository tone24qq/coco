"""Memory-based agent that fuses model predictions with kNN retrieved scores.

This agent demonstrates how a memory bank of prior boards can be used to
refine predictions. Each memory entry stores a hidden-state vector (key) and
model scores (value). At inference time, the query board embedding is compared
with memory keys using cosine similarity. The top-k most similar entries are
used to average the stored scores, which are then fused with the model's own
scores.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import orjson

from dataset import BLANK_VALUE, MASK_TOKEN_ID, validate_board
from utils import ensure_only_blank


def _as_model_input(flat: np.ndarray) -> np.ndarray:
    """Map ``BLANK_VALUE`` to ``MASK_TOKEN_ID`` and ensure int64 array."""
    return np.ascontiguousarray(
        np.where(flat == BLANK_VALUE, MASK_TOKEN_ID, flat).astype(np.int64, copy=False)
    )


def build_memory(
    samples: Sequence[Tuple[np.ndarray, int]],
    model: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a memory bank from ``samples`` using ``model``.

    Parameters
    ----------
    samples:
        Iterable of ``(board, target)`` pairs.
    model:
        Object that provides ``get_hidden_state`` and ``forward`` methods.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(keys, values)`` where ``keys`` has shape ``(M, H)`` and ``values`` has
        shape ``(M, N)``. ``M`` is number of samples, ``H`` is embedding
        dimension and ``N`` equals ``rows*cols``.
    """
    memory_keys: List[np.ndarray] = []
    memory_values: List[np.ndarray] = []
    for board, target in samples:
        flat = board.flatten()
        flat_in = _as_model_input(flat)
        h = model.get_hidden_state(flat_in)
        # 1) 取出原始 logits（可能是 torch.Tensor，且 requires_grad=True）
        logits = model.forward(flat_in)
        # 2) 如果是 Tensor，需先 detach().cpu() 再轉 numpy，避免 requires_grad 錯誤
        if hasattr(logits, "detach"):
            arr = logits.detach().cpu().numpy()
        else:
            arr = np.asarray(logits)
        if arr.ndim == 3:
            arr = arr[0]
        # 3) 從 arr (np.ndarray) 中取 target 分數
        t_idx = int(target) - 1
        if not 0 <= t_idx < arr.shape[1]:
            raise IndexError("target index out of range")
        scores = arr[:, t_idx]
        memory_keys.append(h)
        memory_values.append(scores)
    return np.stack(memory_keys, axis=0), np.stack(memory_values, axis=0)


def predict(
    board: np.ndarray,
    *,
    target: int,
    model: Any,
    memory_keys: np.ndarray,
    memory_values: np.ndarray,
    alpha: float = 0.5,
    k_neighbors: int = 2,
    topk: int | None = None,
    query_index: int | None = None,
) -> List[Dict[str, Any]]:
    """Return ranked blank cell predictions by fusing model and memory scores.

    The function computes cosine similarity between the query board's hidden
    state and each key in ``memory_keys``. The ``k_neighbors`` most similar
    entries are averaged to obtain ``memory`` scores. These scores are blended
    with the model's own predictions using weight ``alpha`` and then ranked.
    """
    validate_board(board, allow_blank=True)
    rows, cols = board.shape
    num_cells = rows * cols

    flat = board.flatten()
    flat_in = _as_model_input(flat)
    h_q = model.get_hidden_state(flat_in)

    norms = np.linalg.norm(memory_keys, axis=1) * np.linalg.norm(h_q)
    cos_sim = (memory_keys @ h_q) / norms
    order = np.argsort(-cos_sim)
    if query_index is not None:
        order = order[order != query_index]
    neighbors = order[:k_neighbors]

    scores_mem = memory_values[neighbors].mean(axis=0)
    t_idx = int(target) - 1
    logits_q = model.forward(flat_in)
    if hasattr(logits_q, "detach"):
        arr = logits_q.detach().cpu().numpy()
    else:
        arr = np.asarray(logits_q)
    if arr.ndim == 3:
        arr = arr[0]
    if not 0 <= t_idx < arr.shape[1]:
        raise IndexError("target index out of range")
    scores_model = arr[:, t_idx]
    scores_final = alpha * scores_model + (1 - alpha) * scores_mem

    mask = flat != BLANK_VALUE
    scores_final[mask] = -np.inf

    coords = [(i // cols, i % cols, scores_final[i]) for i in range(num_cells)]
    coords.sort(key=lambda x: (-x[2], x[0], x[1]))
    if topk is not None:
        coords = coords[:topk]

    results = [{"row": r, "col": c, "score": float(sc)} for r, c, sc in coords]
    filtered = ensure_only_blank(board, results, BLANK_VALUE)
    for item in filtered:
        item["row"] += 1
        item["col"] += 1
    return filtered


def predict_stream(
    board: np.ndarray,
    *,
    target: int,
    model: Any,
    jsonl_path: Path,
    alpha: float = 0.5,
    k_neighbors: int = 2,
    topk: int | None = None,
) -> List[Dict[str, Any]]:
    """Stream kNN retrieval from ``jsonl_path`` and fuse scores.

    Parameters
    ----------
    board:
        Query board. Will not be modified in-place.
    target:
        Target value to predict.
    model:
        Model providing ``get_hidden_state`` and ``forward`` methods.
    jsonl_path:
        Path to newline-delimited JSON archive. Each line must contain a
        ``{"board": ..., "target": ...}`` object. Additional fields are
        ignored.
    alpha:
        Blend factor between model scores and memory scores.
    k_neighbors:
        Number of nearest neighbors to average from the archive.
    topk:
        Number of final predictions to return. ``None`` means all blanks.
    """

    validate_board(board, allow_blank=True)
    rows, cols = board.shape
    num_cells = rows * cols

    flat = board.flatten()
    flat_in = _as_model_input(flat)
    t_idx = int(target) - 1
    logits_q = model.forward(flat_in)
    if hasattr(logits_q, "detach"):
        arr_q = logits_q.detach().cpu().numpy()
    else:
        arr_q = np.asarray(logits_q)
    if arr_q.ndim == 3:
        arr_q = arr_q[0]
    if not 0 <= t_idx < arr_q.shape[1]:
        raise IndexError("target index out of range")
    scores_model = arr_q[:, t_idx]

    q_emb = model.get_hidden_state(flat_in)
    heap: List[Tuple[float, int, np.ndarray]] = []
    with jsonl_path.open("rb") as f:
        for idx, line in enumerate(f):
            item = orjson.loads(line)
            mem_flat = np.asarray(item["board"], dtype=int).flatten()
            mem_in = _as_model_input(mem_flat)
            mem_emb = model.get_hidden_state(mem_in)
            sim = float(
                np.dot(q_emb, mem_emb)
                / (np.linalg.norm(q_emb) * np.linalg.norm(mem_emb) + 1e-8)
            )
            if len(heap) < k_neighbors:
                heapq.heappush(heap, (sim, idx, mem_in))
            elif sim > heap[0][0]:
                heapq.heapreplace(heap, (sim, idx, mem_in))

    if heap:
        mem_scores = np.zeros(num_cells, dtype=float)
        for _, _, mem_in in heap:
            logits = model.forward(mem_in)
            if hasattr(logits, "detach"):
                arr = logits.detach().cpu().numpy()
            else:
                arr = np.asarray(logits)
            if arr.ndim == 3:
                arr = arr[0]
            if not 0 <= t_idx < arr.shape[1]:
                raise IndexError("target index out of range")
            mem_scores += arr[:, t_idx]
        mem_scores /= len(heap)
        scores_final = alpha * scores_model + (1 - alpha) * mem_scores
    else:  # no neighbors found
        scores_final = scores_model

    mask = flat != BLANK_VALUE
    scores_final[mask] = -np.inf

    coords = [(i // cols, i % cols, scores_final[i]) for i in range(num_cells)]
    coords.sort(key=lambda x: (-x[2], x[0], x[1]))
    if topk is not None:
        coords = coords[:topk]
    results = [{"row": r, "col": c, "score": float(sc)} for r, c, sc in coords]
    filtered = ensure_only_blank(board, results, BLANK_VALUE)
    for item in filtered:
        item["row"] += 1
        item["col"] += 1
    return filtered

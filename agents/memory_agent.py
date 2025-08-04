"""Memory-based agent that fuses model predictions with kNN retrieved scores.

This agent demonstrates how a memory bank of prior boards can be used to
refine predictions. Each memory entry stores a hidden-state vector (key) and
model scores (value). At inference time, the query board embedding is compared
with memory keys using cosine similarity. The top-k most similar entries are
used to average the stored scores, which are then fused with the model's own
scores.
"""

from __future__ import annotations

import logging
import mmap
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import orjson

try:  # optional torch dependency
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from dataset import BLANK_VALUE, MASK_TOKEN_ID, validate_board
from utils import ensure_only_blank

logger = logging.getLogger(__name__)


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
        if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            with torch.no_grad():
                logits = model(flat_in)
        else:
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
    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        with torch.no_grad():
            logits_q = model(flat_in)
    else:
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
    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        with torch.no_grad():
            logits_q = model(flat_in)
    else:
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

    batch_size = int(os.getenv("MEMORY_BATCH_SIZE", "64"))
    neighbors = online_topk_from_jsonl(
        jsonl_path=jsonl_path,
        query_board=board,
        model=model,
        top_k=k_neighbors,
        batch_size=batch_size,
    )

    mem_scores = np.zeros(num_cells, dtype=float)
    weight_sum = 0.0
    for rec_id, sim in neighbors:
        rec = load_record_by_id(jsonl_path, rec_id)
        if "scores" in rec:
            scores_vec = np.asarray(rec["scores"], dtype=float)
        else:
            mem_flat = np.asarray(rec["board"], dtype=int).flatten()
            mem_in = _as_model_input(mem_flat)
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    logits = model(mem_in)
            else:
                logits = model.forward(mem_in)
            if hasattr(logits, "detach"):
                arr = logits.detach().cpu().numpy()
            else:
                arr = np.asarray(logits)
            if arr.ndim == 3:
                arr = arr[0]
            if not 0 <= t_idx < arr.shape[1]:
                raise IndexError("target index out of range")
            scores_vec = arr[:, t_idx]
        mem_scores += sim * scores_vec
        weight_sum += sim
    if weight_sum > 0:
        mem_scores /= weight_sum
        scores_final = alpha * scores_model + (1 - alpha) * mem_scores
    else:
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


def _collect_embeddings_from_jsonl(
    jsonl_path: Path, model: Any, batch_size: int
) -> tuple[list[int], np.ndarray]:
    """Return record ids and embeddings for all boards in ``jsonl_path``.

    Records without an ``id`` field will use their line index as the id.
    """

    record_ids: list[int] = []
    embeddings_list: list[np.ndarray] = []
    buf_inputs: list[np.ndarray] = []

    with jsonl_path.open("rb") as f, mmap.mmap(
        f.fileno(), 0, access=mmap.ACCESS_READ
    ) as mm:
        line_idx = 0
        for raw in iter(mm.readline, b""):
            if not raw:
                break
            rec = orjson.loads(raw)
            rec_id = rec.get("id", line_idx)
            record_ids.append(int(rec_id))
            board = np.asarray(rec["board"], dtype=int).flatten()
            buf_inputs.append(_as_model_input(board))
            line_idx += 1
            if len(buf_inputs) >= batch_size:
                batch = np.stack(buf_inputs, axis=0)
                emb = model.get_hidden_state(batch)
                if hasattr(emb, "detach"):
                    emb = emb.detach().cpu().numpy()
                embeddings_list.append(np.atleast_2d(np.asarray(emb)))
                buf_inputs.clear()

        if buf_inputs:
            batch = np.stack(buf_inputs, axis=0)
            emb = model.get_hidden_state(batch)
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            embeddings_list.append(np.atleast_2d(np.asarray(emb)))

    embeddings = (
        np.concatenate(embeddings_list, axis=0) if embeddings_list else np.empty((0, 0))
    )
    return record_ids, embeddings


def online_topk_from_jsonl(
    jsonl_path: str | Path,
    query_board: np.ndarray,
    model: Any,
    top_k: int = 3,
    batch_size: int = 64,
) -> list[tuple[int, float]]:
    """Return ``top_k`` most similar records from ``jsonl_path``.

    The function reads all samples using ``mmap`` and performs batched
    ``get_hidden_state`` calls. Cosine similarities against the query board are
    computed in a single matrix multiplication.
    """

    path = Path(jsonl_path)
    record_ids, embeddings = _collect_embeddings_from_jsonl(path, model, batch_size)
    if not record_ids:
        return []

    q_flat = _as_model_input(np.asarray(query_board, dtype=int).flatten())
    q_emb = model.get_hidden_state(q_flat)
    if hasattr(q_emb, "detach"):
        q_emb = q_emb.detach().cpu().numpy()
    q_norm = np.linalg.norm(q_emb)

    norms = np.linalg.norm(embeddings, axis=1)
    sims = embeddings @ q_emb
    sims /= norms * q_norm + 1e-8

    k = min(top_k, len(sims))
    idx_topk = np.argpartition(-sims, k - 1)[:k]
    idx_topk = idx_topk[np.argsort(-sims[idx_topk])]
    return [(record_ids[i], float(sims[i])) for i in idx_topk]


def online_topk_from_directory(
    dir_path: str | Path,
    query_board: np.ndarray,
    model: Any,
    top_k: int = 3,
    batch_size: int = 64,
) -> list[tuple[int, float]]:
    """Return ``top_k`` most similar records from all JSONL files under ``dir_path``."""

    record_ids: list[int] = []
    embeddings_list: list[np.ndarray] = []
    for jsonl_path in Path(dir_path).rglob("*.jsonl"):
        ids, emb = _collect_embeddings_from_jsonl(jsonl_path, model, batch_size)
        if ids:
            record_ids.extend(ids)
            embeddings_list.append(emb)

    if not record_ids:
        return []

    embeddings = np.concatenate(embeddings_list, axis=0)
    q_flat = _as_model_input(np.asarray(query_board, dtype=int).flatten())
    q_emb = model.get_hidden_state(q_flat)
    if hasattr(q_emb, "detach"):
        q_emb = q_emb.detach().cpu().numpy()
    q_norm = np.linalg.norm(q_emb)

    norms = np.linalg.norm(embeddings, axis=1)
    sims = embeddings @ q_emb
    sims /= norms * q_norm + 1e-8

    k = min(top_k, len(sims))
    idx_topk = np.argpartition(-sims, k - 1)[:k]
    idx_topk = idx_topk[np.argsort(-sims[idx_topk])]
    return [(record_ids[i], float(sims[i])) for i in idx_topk]


def load_record_by_id(jsonl_path: Path, rec_id: int) -> Dict[str, Any]:
    """Load a record with ``rec_id`` from ``jsonl_path``.

    If the JSON lines do not contain an explicit ``id`` field, ``rec_id`` is
    treated as the zero-based line index.
    """

    with jsonl_path.open("rb") as f:
        for idx, raw in enumerate(f):
            rec = orjson.loads(raw)
            cur_id = rec.get("id", idx)
            if cur_id == rec_id:
                return rec
    raise KeyError(f"record id {rec_id} not found in {jsonl_path}")

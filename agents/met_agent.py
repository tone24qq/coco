"""Agent that predicts scratch card positions using a DynamicMET model."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from dataset import BLANK_VALUE, MASK_TOKEN_ID, validate_board
from model import DynamicMET
from utils import ensure_only_blank, index_to_coord

logger = logging.getLogger(__name__)


def predict(
    board: np.ndarray,
    *,
    target: int,
    model: DynamicMET,
    topk: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Predict blank positions for ``target`` using ``model``.

    The function scores **all** cells on the board in a single forward pass
    and then selects the scores corresponding to blank cells.  Returned
    results are sorted by score in descending order.  ``topk`` can be used to
    limit the number of returned positions (e.g. ``topk=3`` for top-3).
    """
    validate_board(board, allow_blank=True)
    rows, cols = board.shape
    flat = board.flatten()

    candidate_idx = np.where(flat == BLANK_VALUE)[0]
    if candidate_idx.size == 0:
        return []

    arr_inp = np.where(flat == BLANK_VALUE, MASK_TOKEN_ID, flat).astype(np.int64)

    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        inp = torch.as_tensor(arr_inp, dtype=torch.long).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            logits = model(inp)
        n = model.num_fields
        logger.info(
            "IDX semantics: 0=blank(ignored), 1..%s=numbers 1..%s ; logits.shape=%s",
            n,
            n,
            tuple(logits.shape),
        )
        probs = torch.softmax(logits, dim=-1)
        V = probs.shape[-1]
        target_idx = target
        if not (0 <= target_idx < V):
            raise RuntimeError(
                f"target_idx out of range: target={target} -> {target_idx}, V={V}"
            )
        scores_all = probs[0, :, target_idx].detach().cpu().numpy()
    else:
        inp = arr_inp.reshape(1, -1)
        logits = model(inp)
        n = model.num_fields
        logger.info(
            "IDX semantics: 0=blank(ignored), 1..%s=numbers 1..%s ; logits.shape=%s",
            n,
            n,
            tuple(logits.shape),
        )
        arr = np.asarray(logits)
        V = arr.shape[-1]
        target_idx = target
        if not (0 <= target_idx < V):
            raise RuntimeError(
                f"target_idx out of range: target={target} -> {target_idx}, V={V}"
            )
        scores_all = arr[0, :, target_idx]

    candidate_scores = scores_all[candidate_idx]
    assert (
        candidate_scores.shape[0] == candidate_idx.shape[0]
    ), f"空白格 {candidate_idx.shape[0]} 个，却只打分了 {candidate_scores.shape[0]} 个！"
    logger.debug("✅ 完整打分：空白格共 %s 个，已收到分数", candidate_idx.shape[0])
    order = np.argsort(-candidate_scores)
    top5_idx = order[: min(5, candidate_scores.size)]
    logger.debug(
        "Top-5 raw scores: %s",
        ", ".join(f"{candidate_scores[i]:.4f}" for i in top5_idx),
    )
    top_indices = candidate_idx[order]
    if topk is not None:
        top_indices = top_indices[:topk]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        r, c = index_to_coord(int(idx), board.shape)
        results.append({"row": r, "col": c, "score": float(scores_all[idx])})
    results = ensure_only_blank(board, results, BLANK_VALUE)
    for item in results:
        item["row"] += 1
        item["col"] += 1
    return results

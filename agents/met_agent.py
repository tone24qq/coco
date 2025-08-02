"""Agent that predicts scratch card positions using a DynamicMET model."""

import logging
from typing import Any, Dict, List

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from dataset import BLANK_VALUE, MASK_TOKEN_ID
from model import DynamicMET
from utils import ensure_only_blank, index_to_coord

logger = logging.getLogger(__name__)


def predict(
    board: np.ndarray, *, target: int, model: DynamicMET, topk: int = 3
) -> List[Dict[str, Any]]:
    """Predict top-k blank positions for ``target`` using ``model``."""
    rows, cols = board.shape
    flat = board.flatten()

    candidate_idx = np.where(flat == BLANK_VALUE)[0]
    if candidate_idx.size == 0:
        return []

    arr_inp = np.where(flat == BLANK_VALUE, MASK_TOKEN_ID, flat).astype(np.int64)

    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            inp = torch.as_tensor(arr_inp, dtype=torch.long).unsqueeze(0)
            logits = model(inp)
            logger.info(
                "IDX semantics: 0=blank(ignored), 1..80=numbers 1..80 ; logits.shape=%s",
                tuple(logits.shape),
            )
            probs = torch.softmax(logits, dim=-1)
            scores_all = probs[0, :, target].detach().cpu().numpy()
    else:
        inp = arr_inp.reshape(1, -1)
        logits = model(inp)
        logger.info(
            "IDX semantics: 0=blank(ignored), 1..80=numbers 1..80 ; logits.shape=%s",
            np.shape(logits),
        )
        arr = np.asarray(logits)
        exp_arr = np.exp(arr - arr.max(axis=-1, keepdims=True))
        probs = exp_arr / exp_arr.sum(axis=-1, keepdims=True)
        scores_all = probs[0, :, target]

    candidate_scores = scores_all[candidate_idx]
    k = min(topk, candidate_scores.size)
    logger.info("TopK: candidates=%s, k=%s", candidate_idx.size, k)
    if k == 0:
        return []
    topk_local = np.argpartition(candidate_scores, -k)[-k:]
    order = np.lexsort((candidate_idx[topk_local], -candidate_scores[topk_local]))
    top_indices = candidate_idx[topk_local][order][-k:][::-1]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        r, c = index_to_coord(int(idx), board.shape)
        results.append({"row": r, "col": c, "score": float(scores_all[idx])})
    results = ensure_only_blank(board, results, BLANK_VALUE)
    for item in results:
        item["row"] += 1
        item["col"] += 1
    return results

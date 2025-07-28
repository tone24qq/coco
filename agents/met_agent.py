"""Agent that predicts scratch card positions using a DynamicMET model."""

from typing import Any, Dict, List

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from dataset import BLANK_VALUE
from model import DynamicMET
from utils import ensure_only_blank, index_to_coord


def predict(
    board: np.ndarray, *, target: int, model: DynamicMET, topk: int = 3
) -> List[Dict[str, Any]]:
    """Predict top-k blank positions for ``target`` using ``model``."""
    rows, cols = board.shape
    flat = board.flatten()

    mask_pos = np.where(flat == BLANK_VALUE)[0]
    if mask_pos.size == 0:
        return []

    arr_inp = np.where(flat < 0, 0, flat)

    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        inp = torch.as_tensor(arr_inp, dtype=torch.long).unsqueeze(0)
        logits = model(inp)
        probs = torch.softmax(logits, dim=-1)
        V = probs.shape[-1]
        target_idx = target if V == flat.size + 1 else target - 1
        if not (0 <= target_idx < V):
            raise RuntimeError(
                f"target_idx out of range: target={target} -> {target_idx}, V={V}"
            )
        scores_all = probs[0, :, target_idx].detach().cpu().numpy()
    else:
        inp = arr_inp.reshape(1, -1).astype(np.int64, copy=False)
        logits = model(inp)
        arr = np.asarray(logits)
        V = arr.shape[-1]
        target_idx = target if V == flat.size + 1 else target - 1
        if not (0 <= target_idx < V):
            raise RuntimeError(
                f"target_idx out of range: target={target} -> {target_idx}, V={V}"
            )
        scores_all = arr[0, :, target_idx]

    cand_scores = scores_all[mask_pos]
    k = min(topk, cand_scores.size)
    if k == 0:
        return []
    local_idx = np.argsort(cand_scores)[-k:][::-1]
    top_indices = mask_pos[local_idx]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        r, c = index_to_coord(int(idx), board.shape)
        results.append({"row": r, "col": c, "score": float(scores_all[idx])})
    results = ensure_only_blank(board, results, BLANK_VALUE)
    for item in results:
        item["row"] += 1
        item["col"] += 1
    return results

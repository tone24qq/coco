"""Agent that predicts scratch card positions using a DynamicMET model."""

from typing import Any, Dict, List

import numpy as np
import torch

from model import DynamicMET


def predict(
    board: np.ndarray, target: int, *, model: DynamicMET, topk: int = 3
) -> List[Dict[str, Any]]:
    """Predict top-k positions for ``target`` using ``model``.

    Parameters
    ----------
    board : np.ndarray
        Board array containing integers. ``-1`` denotes empty cells.
    target : int
        Target value to locate.
    model : DynamicMET
        Pretrained model used for prediction.
    topk : int, optional
        Number of positions to return, by default ``3``.

    Returns
    -------
    List[Dict[str, Any]]
        Each dictionary contains ``row``, ``col`` and ``score`` fields.
    """

    shape = board.shape
    board_proc = np.where(board < 0, 0, board).astype(int)
    inp = torch.tensor(board_proc.flatten()).long().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=-1)[0, :, target]
        scores, indices = torch.topk(probs, k=topk)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        r, c = divmod(idx, shape[1])
        results.append({"row": r, "col": c, "score": float(score)})
    return results

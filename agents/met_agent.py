"""Agent that predicts scratch card positions using a DynamicMET model."""

from typing import Any, Dict, List

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

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

    if TORCH_AVAILABLE:
        inp = torch.tensor(board_proc.flatten()).long().unsqueeze(0)
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=-1)[0, :, target]
            scores, indices = torch.topk(probs, k=topk)
        scores_list = scores.tolist()
        indices_list = indices.tolist()
    else:
        inp = board_proc.flatten().reshape(1, -1)
        logits = model(inp)
        probs = np.asarray(logits)[0, :, target]
        indices_list = np.argsort(probs)[-topk:][::-1].tolist()
        scores_list = probs[indices_list].tolist()

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores_list, indices_list):
        r, c = divmod(int(idx), shape[1])
        results.append({"row": r, "col": c, "score": float(score)})
    return results

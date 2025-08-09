from __future__ import annotations

from typing import Dict, List

import torch


def compute_topk_positions(
    probs: torch.Tensor, tokens: torch.Tensor, query_num: int, k: int, cols: int
) -> List[Dict[str, float]]:
    """Compute top-k positions for ``query_num`` on masked tokens.

    Parameters
    ----------
    probs: torch.Tensor
        Probability tensor of shape ``[L, V]``.
    tokens: torch.Tensor
        Token tensor of shape ``[L]`` where ``0`` indicates a hole.
    query_num: int
        The number to query.
    k: int
        Return up to ``k`` positions.
    cols: int
        Number of columns in the board for converting flat index.
    """
    holes = tokens == 0
    num_holes = int(holes.sum().item())
    if num_holes == 0:
        return []
    p_num = probs[:, query_num]
    p_masked = torch.where(holes, p_num, torch.full_like(p_num, float("-inf")))
    k = min(k, num_holes)
    vals, idxs = torch.topk(p_masked, k)
    topk = []
    for v, idx in zip(vals.tolist(), idxs.tolist()):
        r, c = divmod(idx, cols)
        topk.append({"row": r, "col": c, "prob": float(probs[idx, query_num].item())})
    return topk

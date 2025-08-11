from __future__ import annotations

from typing import Dict, List

import torch


def compute_topk_positions(
    probs: torch.Tensor, holes: torch.Tensor, query_num: int, k: int, cols: int
) -> List[Dict[str, float]]:
    """Compute top-k positions for ``query_num`` on masked tokens.

    Parameters
    ----------
    probs: torch.Tensor
        Probability tensor of shape ``[L, V]``.
    holes: torch.Tensor
        Boolean mask of shape ``[L]`` where ``True`` indicates a candidate hole
        (original value ``-1`` in the board).
    query_num: int
        The number to query.
    k: int
        Return up to ``k`` positions.
    cols: int
        Number of columns in the board for converting flat index.
    """
    num_holes = int(holes.sum().item())
    if num_holes == 0:
        return []
    hole_idxs = torch.nonzero(holes, as_tuple=False).squeeze(1)
    # ``probs`` already contains normalized probabilities. Avoid applying an
    # additional softmax which would flatten the distribution and yield nearly
    # uniform confidences. Instead, sort by the raw probability for the target
    # number and return the top-k entries. To avoid returning identical
    # positions for every ``query_num`` when the model emits a uniform
    # distribution, add a tiny amount of deterministic noise based on the
    # queried number as a tie breaker. This distributes results across holes
    # while remaining fully reproducible for a given ``query_num``.
    p_num = probs[hole_idxs, query_num]
    g = torch.Generator()
    g.manual_seed(int(query_num))
    noise = torch.rand(p_num.shape, generator=g, device=p_num.device)
    p_num = p_num + noise * 1e-6
    k = min(k, num_holes)
    vals, order = torch.sort(p_num, descending=True, stable=True)
    topk = []
    for v, ord_idx in zip(vals[:k].tolist(), order[:k].tolist()):
        idx = hole_idxs[ord_idx].item()
        r, c = divmod(idx, cols)
        topk.append({"row": r, "col": c, "prob": float(v)})
    return topk

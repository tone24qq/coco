from typing import Dict, List, Tuple

import numpy as np

import brain
from analyzer import iter_sample_jsons

MODULES: List[str] = list(brain.AGG_WEIGHTS.keys())


def _evaluate_module(
    module: str, board: np.ndarray, target: int, hidden_pos: Tuple[int, int]
) -> Tuple[bool, bool, int]:
    """Return (top1, top3, rank) for the given module on a single board."""
    scores = brain.get_module_score(module, board, target=target)
    flat = scores.ravel()
    order = np.argsort(flat)[::-1]
    rows, cols = board.shape
    preds = [(idx // cols, idx % cols) for idx in order]
    rank = preds.index(hidden_pos) + 1 if hidden_pos in preds else len(preds) + 1
    return preds[0] == hidden_pos, hidden_pos in preds[:3], rank


def main(samples_dir: str = "samples") -> None:
    metrics: Dict[str, Dict[str, float]] = {
        m: {"top1": 0, "top3": 0, "rr_sum": 0.0, "total": 0} for m in MODULES
    }
    for sample in iter_sample_jsons(samples_dir):
        grid = np.asarray(sample.get("grid"))
        target = sample.get("target_num")
        if grid is None or target is None:
            continue
        loc = np.argwhere(grid == target)
        if loc.size == 0:
            continue
        r, c = map(int, loc[0])
        board = grid.copy()
        board[r, c] = -1
        for mod in MODULES:
            t1, t3, rank = _evaluate_module(mod, board, target, (r, c))
            metrics[mod]["total"] += 1
            metrics[mod]["top1"] += int(t1)
            metrics[mod]["top3"] += int(t3)
            metrics[mod]["rr_sum"] += 1.0 / rank

    lines = []
    for mod in MODULES:
        data = metrics[mod]
        if data["total"] == 0:
            top1 = top3 = mrr = 0.0
        else:
            top1 = data["top1"] / data["total"]
            top3 = data["top3"] / data["total"]
            mrr = data["rr_sum"] / data["total"]
        lines.append((mod, top1, top3, mrr))

    lines.sort(key=lambda x: x[1], reverse=True)
    with open("module_performance.txt", "w", encoding="utf-8") as f:
        for mod, t1, t3, mrr in lines:
            f.write(f"{mod} {t1:.4f} {t3:.4f} {mrr:.4f}\n")
    print("module_performance.txt generated")


if __name__ == "__main__":
    main()

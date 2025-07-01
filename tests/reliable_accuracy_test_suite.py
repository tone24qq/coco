import numpy as np

from analyzer import probability_heatmap
from modules import generate_unique_grid


def run_accuracy_trials(n_rounds: int = 1000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_rounds):
        rows = int(rng.integers(4, 21))
        cols = int(rng.integers(5, 21))
        full = generate_unique_grid(rows, cols, rng=rng)
        mask = rng.random((rows, cols)) < 0.5
        board = full.copy()
        board[mask] = -1
        blanks = np.argwhere(board == -1)
        if blanks.size == 0:
            continue
        r, c = blanks[rng.integers(len(blanks))]
        target = int(full[r, c])
        heat = probability_heatmap(
            board, target, n_iter=1000, seed=int(rng.integers(1_000_000))
        )
        pr, pc = divmod(int(np.argmax(heat)), cols)
        if pr == r and pc == c:
            hits += 1
    return hits / float(n_rounds)

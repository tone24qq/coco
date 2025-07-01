import numpy as np

from analyzer import probability_heatmap
from modules import generate_unique_grid


def _simulate_single(
    rng: np.random.Generator, rows: int, cols: int, mask_ratio: float
) -> bool:
    full = generate_unique_grid(rows, cols, rng=rng)
    mask = rng.random((rows, cols)) < mask_ratio
    board = full.copy()
    board[mask] = -1
    blanks = np.argwhere(board == -1)
    if blanks.size == 0:
        return False
    r, c = blanks[rng.integers(len(blanks))]
    target = int(full[r, c])
    heat = probability_heatmap(
        board, target, n_iter=1000, seed=int(rng.integers(1_000_000))
    )
    pr, pc = divmod(int(np.argmax(heat)), cols)
    return pr == r and pc == c


def run_infinite_test(
    min_size: int = 4,
    max_size: int = 20,
    mask_ratio: float = 0.5,
    max_iters: int = 1_000_000,
    seed: int = 42,
    log_every: int = 1000,
) -> None:
    """Continuous stress test printing accuracy periodically."""
    rng = np.random.default_rng(seed)
    hits = total = 0
    for i in range(1, max_iters + 1):
        rows = int(rng.integers(min_size, max_size + 1))
        cols = int(rng.integers(max(min_size, 5), max_size + 1))
        if _simulate_single(rng, rows, cols, mask_ratio):
            hits += 1
        total += 1
        if i % log_every == 0:
            acc = hits / float(total)
            print(f"{i}: accuracy={acc:.3f}")


def run_until_converged(
    min_size: int = 4,
    max_size: int = 20,
    mask_ratio: float = 0.5,
    batch_size: int = 200,
    delta: float = 0.02,
    z: float = 1.96,
    seed: int = 0,
) -> tuple[float, float, int]:
    """Run batches of simulations until the confidence interval is below ``delta``."""
    rng = np.random.default_rng(seed)
    hits = total = 0
    max_batches = 50
    for _ in range(max_batches):
        for _ in range(batch_size):
            rows = int(rng.integers(min_size, max_size + 1))
            cols = int(rng.integers(max(min_size, 5), max_size + 1))
            if _simulate_single(rng, rows, cols, mask_ratio):
                hits += 1
            total += 1
        p = hits / float(total)
        hw = z * ((p * (1 - p) / max(1, total)) ** 0.5)
        if hw <= delta and total >= batch_size * 5:
            break
    return p, hw, total

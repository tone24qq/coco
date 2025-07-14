from typing import Any, Dict, List

import numpy as np


def _matrix_factorization(
    board: np.ndarray,
    rank: int = 3,
    lr: float = 0.005,
    max_iter: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Fill missing values using simple matrix factorization.

    Args:
        board: 2D array with -1 as missing entries.
        rank: latent rank for factorization.
        lr: learning rate for gradient descent.
        max_iter: maximum number of iterations.
        seed: random seed for reproducibility.

    Returns:
        Completed board with integers in range 0-99.
    """
    rng = np.random.default_rng(seed)
    mask = board != -1
    m, n = board.shape

    # initial guess: row mean
    row_sum = np.where(mask, board, 0).sum(axis=1)
    row_count = mask.sum(axis=1)
    row_mean = np.divide(
        row_sum,
        row_count,
        out=np.zeros_like(row_sum, dtype=float),
        where=row_count != 0,
    )
    filled = board.astype(float).copy()
    for i in range(m):
        for j in range(n):
            if not mask[i, j]:
                filled[i, j] = row_mean[i]

    U = rng.normal(scale=0.1, size=(m, rank))
    V = rng.normal(scale=0.1, size=(rank, n))
    for _ in range(max_iter):
        pred = U @ V
        error = (pred - filled) * mask
        grad_U = error @ V.T
        grad_V = U.T @ error
        U -= lr * grad_U
        V -= lr * grad_V
        if np.linalg.norm(error) < 1e-3:
            break
    result = np.clip(np.round(U @ V), 0, 99).astype(int)
    return result


def predict(board: np.ndarray, target: int, **kwargs: Any) -> List[Dict[str, Any]]:
    """Predict candidate cells that may contain the target value.

    The board is first completed via matrix factorization. Candidate cells are
    scored by proximity between predicted value and target.
    """
    completed = _matrix_factorization(board, **kwargs)
    diff = np.abs(completed - target)
    flat_indices = np.argsort(diff, axis=None)
    results: List[Dict[str, Any]] = []
    for idx in flat_indices:
        r, c = divmod(idx, completed.shape[1])
        score = float(1.0 / (1.0 + diff[r, c]))
        results.append({"row": int(r), "col": int(c), "score": score})
    return results

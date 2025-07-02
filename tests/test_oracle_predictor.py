import time

import numpy as np

from modules import generate_unique_grid
from oracle_predictor import predict_target_location


def test_oracle_accuracy_and_latency():
    rng = np.random.default_rng(0)
    rounds = 20
    hits = 0
    durations = []
    for _ in range(rounds):
        rows = int(rng.integers(4, 21))
        cols = int(rng.integers(4, 21))
        full = generate_unique_grid(rows, cols, rng=rng)
        mask = rng.random((rows, cols)) < rng.uniform(0.4, 0.6)
        board = full.copy()
        board[mask] = -1
        blanks = np.argwhere(board == -1)
        if blanks.size == 0:
            continue
        r, c = blanks[rng.integers(len(blanks))]
        target = int(full[r, c])
        start = time.perf_counter()
        pred = predict_target_location(board.tolist(), target)
        durations.append(time.perf_counter() - start)
        if pred["row"] == r and pred["col"] == c:
            hits += 1
    acc = hits / rounds
    assert acc >= 0.99
    assert max(durations) <= 1.0

from pathlib import Path

import numpy as np

import app
from dataset import BLANK_VALUE


def test_find_similar_board(tmp_path):
    rows, cols = 2, 2
    boards = np.array(
        [
            [[1, 2], [3, 4]],
            [[1, 2], [4, 3]],
            [[4, 3], [2, 1]],
        ],
        dtype=int,
    )
    keys = np.zeros((boards.shape[0], 1), dtype=float)
    values = np.zeros((boards.shape[0], rows * cols), dtype=float)
    targets = np.ones(boards.shape[0], dtype=int)
    path = Path("data_archives") / "2x2_memory.npz"
    np.savez(
        path,
        keys=keys,
        values=values,
        targets=targets,
        boards=boards.reshape(boards.shape[0], -1),
    )
    try:
        app._load_memory_for_shape(rows, cols)
        query = boards[0].copy()
        query[1, 1] = BLANK_VALUE
        sims = app.find_similar_board(rows, cols, query, k=2)
        assert sims[0]["sample_idx"] == 0
        assert sims[0]["distance"] == 0
        assert len(sims) == 2
    finally:
        path.unlink()
        app.memories.pop((rows, cols), None)
        app.memory_targets.pop((rows, cols), None)
        app.memory_boards.pop((rows, cols), None)
        app.hnsw_indices.pop((rows, cols), None)
        app.hnsw_board_indices.pop((rows, cols), None)

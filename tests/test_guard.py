import numpy as np

from utils.guard import ensure_only_blank, ensure_unique, index_to_coord


def test_index_to_coord_matches_numpy() -> None:
    shape = (4, 5)
    for idx in range(shape[0] * shape[1]):
        assert index_to_coord(idx, shape) == tuple(np.unravel_index(idx, shape))


def test_ensure_only_blank_and_unique() -> None:
    board = np.array([[1, -1], [-1, 2]])
    preds = [
        {"row": 0, "col": 1, "score": 0.5},
        {"row": 1, "col": 0, "score": 0.3},
        {"row": 0, "col": 1, "score": 0.7},  # duplicate
        {"row": 0, "col": 0, "score": 0.1},  # non blank
    ]
    filtered = ensure_only_blank(board, preds)
    assert len(filtered) == 3  # non blank removed
    unique = ensure_unique(filtered)
    assert len(unique) == 2
    coords = {(p["row"], p["col"]) for p in unique}
    assert coords == {(0, 1), (1, 0)}

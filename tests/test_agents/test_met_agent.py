import numpy as np

from agents.met_agent import predict
from dataset import BLANK_VALUE
from model import DynamicMET


def test_met_agent_predict_on_10x12() -> None:
    rng = np.random.default_rng(42)
    rows, cols = 10, 12
    grid = np.arange(1, rows * cols + 1).reshape(rows, cols)
    blank_indices = rng.choice(rows * cols, size=rng.integers(15, 26), replace=False)
    for idx in blank_indices:
        r, c = divmod(idx, cols)
        grid[r, c] = BLANK_VALUE
    non_blanks = np.argwhere(grid != BLANK_VALUE)
    target_r, target_c = non_blanks[rng.integers(len(non_blanks))]
    target = int(grid[target_r, target_c])
    model = DynamicMET(rows * cols, num_values=rows * cols, rows=rows, cols=cols)
    result = predict(grid.copy(), target=target, model=model)
    assert isinstance(result, list)
    blank_count = np.sum(grid == BLANK_VALUE)
    assert len(result) == blank_count
    scores = [item["score"] for item in result]
    assert scores == sorted(scores, reverse=True)


def test_met_agent_only_returns_blank() -> None:
    rows, cols = 4, 5
    board = np.array(
        [
            [1, BLANK_VALUE, 3, 4, 5],
            [6, 7, BLANK_VALUE, 9, 10],
            [11, 12, 13, BLANK_VALUE, 15],
            [16, BLANK_VALUE, 18, 19, 20],
        ]
    )
    model = DynamicMET(rows * cols, num_values=rows * cols, rows=rows, cols=cols)
    target = 15
    res = predict(board.copy(), target=target, model=model, topk=3)
    blank_count = np.sum(board == BLANK_VALUE)
    assert len(res) == min(3, blank_count)
    for item in res:
        r0, c0 = item["row"] - 1, item["col"] - 1
        assert board[r0, c0] == BLANK_VALUE


def test_met_agent_scoring_and_order() -> None:
    rng = np.random.default_rng(123)
    rows, cols = rng.integers(4, 8), rng.integers(4, 8)
    vals = np.arange(1, rows * cols + 1)
    rng.shuffle(vals)
    grid = vals.reshape(rows, cols)
    blank_indices = rng.choice(
        rows * cols, size=rng.integers(1, rows * cols // 2), replace=False
    )
    for idx in blank_indices:
        r, c = divmod(idx, cols)
        grid[r, c] = BLANK_VALUE
    non_blanks = np.argwhere(grid != BLANK_VALUE)
    target_r, target_c = non_blanks[rng.integers(len(non_blanks))]
    target = int(grid[target_r, target_c])
    model = DynamicMET(rows * cols, num_values=rows * cols, rows=rows, cols=cols)
    results = predict(grid.copy(), target=target, model=model)
    blank_count = np.sum(grid == BLANK_VALUE)
    assert len(results) == blank_count
    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True)


def test_met_agent_stable_order_with_ties() -> None:
    class DummyModel:
        def __init__(self) -> None:
            self.num_fields = 4
            self.num_values = 4
            self.rows = 2
            self.cols = 2

        def __call__(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - stub
            batch, n = x.shape
            return np.zeros((batch, n, self.num_values))

        def eval(self) -> None:  # pragma: no cover - compatibility stub
            pass

    board = np.full((2, 2), BLANK_VALUE)
    model = DummyModel()
    res = predict(board.copy(), target=1, model=model, topk=3)
    coords = [(item["row"], item["col"]) for item in res]
    assert coords == [(1, 1), (1, 2), (2, 1)]

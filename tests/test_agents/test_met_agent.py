import numpy as np

from agents.met_agent import predict
from dataset import BLANK_VALUE
from model import DynamicMET


def test_met_agent_predict_on_10x12() -> None:
    rng = np.random.default_rng(42)
    rows, cols = 10, 12
    grid = rng.integers(1, 100, size=(rows, cols))
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
    assert len(result) > 0
    for item in result:
        assert isinstance(item, dict)
        assert "row" in item and "col" in item and "score" in item


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
    assert len(res) <= 3
    for item in res:
        r0, c0 = item["row"] - 1, item["col"] - 1
        assert board[r0, c0] == BLANK_VALUE

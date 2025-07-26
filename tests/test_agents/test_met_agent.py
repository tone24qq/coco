import numpy as np

from agents.met_agent import predict
from model import DynamicMET


def test_met_agent_predict_on_10x12() -> None:
    rng = np.random.default_rng(42)
    rows, cols = 10, 12
    grid = rng.integers(1, 100, size=(rows, cols))
    blank_indices = rng.choice(rows * cols, size=rng.integers(15, 26), replace=False)
    for idx in blank_indices:
        r, c = divmod(idx, cols)
        grid[r, c] = -1
    non_blanks = np.argwhere(grid != -1)
    target_r, target_c = non_blanks[rng.integers(len(non_blanks))]
    target = int(grid[target_r, target_c])
    model = DynamicMET(rows * cols, 100)
    result = predict(grid.copy(), target=target, model=model)
    assert isinstance(result, list)
    assert len(result) > 0
    for item in result:
        assert isinstance(item, dict)
        assert "row" in item and "col" in item and "score" in item

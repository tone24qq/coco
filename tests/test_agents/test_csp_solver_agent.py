import numpy as np

from agents.csp_solver_agent import predict, solve


def _is_valid(board: np.ndarray) -> bool:
    n = board.shape[0]
    digits = set(range(1, n + 1))
    for i in range(n):
        if set(board[i]) != digits:
            return False
        if set(board[:, i]) != digits:
            return False
    return True


def test_csp_solver_on_simple_board(board_and_solution):
    board, _solution = board_and_solution
    solved = solve(board)
    assert solved is not None
    assert _is_valid(solved)
    rng = np.random.default_rng(0)
    target = int(solved.flat[rng.integers(solved.size)])
    preds = predict(board, target=target)
    solved_positions = set(zip(*np.where(solved == target)))
    assert {(p["row"], p["col"]) for p in preds} == solved_positions


def test_csp_solver_predict_interface():
    rng = np.random.default_rng(42)
    rows, cols = 4, 4
    board = rng.integers(1, 5, size=(rows, cols))
    blank_indices = rng.choice(rows * cols, size=5, replace=False)
    for idx in blank_indices:
        r, c = divmod(idx, cols)
        board[r, c] = -1
    non_blanks = np.argwhere(board != -1)
    target_r, target_c = non_blanks[rng.integers(len(non_blanks))]
    target = board[target_r, target_c]
    result = predict(board.copy(), target=target)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, dict)
        assert "row" in item and "col" in item and "score" in item

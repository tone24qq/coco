from agents.csp_solver_agent import solve


def test_solver_beats_official(board_and_solution):
    board, _ = board_and_solution
    solved = solve(board.copy())
    assert solved is not None
    digits = set(range(1, solved.shape[0] + 1))
    for i in range(solved.shape[0]):
        assert set(solved[i]) == digits
        assert set(solved[:, i]) == digits

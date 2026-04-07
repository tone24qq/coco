from src.board_solver import solve_board
from src.ticket_specs import get_ticket_spec


def test_solver_repairs_duplicate_using_candidates() -> None:
    spec = get_ticket_spec("20")
    grid = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 19]]
    candidates = {(4, 3): [20, 19]}
    labels = {(4, 3): "printed_number"}
    out = solve_board(grid, candidates, labels, spec)
    assert out.grid[4][3] == 20


def test_solver_pending_when_ambiguous() -> None:
    spec = get_ticket_spec("20")
    grid = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, None, None]]
    candidates = {(4, 2): [19, 20], (4, 3): [19, 20]}
    labels = {(4, 2): "printed_number", (4, 3): "printed_number"}
    out = solve_board(grid, candidates, labels, spec)
    assert len(out.pending_cells) >= 1

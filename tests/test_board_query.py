from src.board_query import build_value_to_position, find_number_positions


def test_value_to_position_ok() -> None:
    grid = [[1, None], [2, 3]]
    out = build_value_to_position(grid)
    assert out["1"][0]["row_1based"] == 1
    assert out["3"][0]["col_1based"] == 2


def test_find_number_not_found() -> None:
    grid = [[1, 2], [3, 4]]
    out = find_number_positions(grid, 9)
    assert out["status"] == "not_found"
    assert out["positions"] == []


def test_find_number_duplicate_violation() -> None:
    grid = [[1, 2], [2, 4]]
    out = find_number_positions(grid, 2)
    assert out["contract_violation"] is True
    assert len(out["positions"]) == 2

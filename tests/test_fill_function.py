from app import fill_cell


def test_fill_success() -> None:
    board = [[-1, 2], [3, -1]]
    result = fill_cell(board, target=1, row=0, col=0)
    assert result == [[1, 2], [3, -1]]


def test_fill_cell_not_blank() -> None:
    board = [[4, -1], [-1, 2]]
    try:
        fill_cell(board, target=3, row=0, col=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_fill_target_exists() -> None:
    board = [[-1, 1], [-1, -1]]
    try:
        fill_cell(board, target=1, row=0, col=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

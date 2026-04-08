from src.number_position_predictor import predict_number_positions


def test_predictor_exact_found() -> None:
    grid = [[1, 2], [3, 4]]
    out = predict_number_positions(grid, 3, [], [], [])
    assert out["query_status"] == "exact_found"


def test_predictor_missing_returns_top5() -> None:
    grid = [[1, None], [3, 4]]
    out = predict_number_positions(grid, 2, [2], [{"row": 0, "col": 1}], [])
    assert out["query_status"] == "predicted"
    assert len(out["top5_position_candidates"]) >= 1


def test_predictor_black_excluded() -> None:
    grid = [[None, None], [3, 4]]
    out = predict_number_positions(grid, 1, [1, 2], [], [{"row": 1, "col": 1}])
    assert all(
        not (x["row_1based"] == 1 and x["col_1based"] == 1)
        for x in out["top5_position_candidates"]
    )

import modern_predictor


def test_predict_location_basic():
    grid = [[1, -1], [2, -1]]
    preds = modern_predictor.predict_location(grid)
    assert isinstance(preds, list) and len(preds) == 2
    assert preds[0]["score"] >= preds[1]["score"]
    for p in preds:
        assert 0 <= p["row"] < 2
        assert 0 <= p["col"] < 2


def test_predict_location_dynamic():
    grid = [[1, -1], [2, -1]]
    preds = modern_predictor.predict_location(grid, rank_method="dynamic")
    assert len(preds) == 2


def test_predict_location_borda():
    grid = [[1, -1], [2, -1]]
    preds = modern_predictor.predict_location(grid, rank_method="borda")
    assert len(preds) == 2

from analyzer import predict_scratch_card


def test_predictions_only_blank_cells():
    grid = [
        [-1, 2, 3],
        [4, -1, 6],
        [7, 8, 9],
    ]
    out = predict_scratch_card(grid, target_num=1, result_top_k=3)
    blanks = {(0, 0), (1, 1)}
    for p in out["predictions"]:
        assert (p["row"], p["col"]) in blanks

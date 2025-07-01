import analyzer


def test_compute_position_probabilities():
    probs = analyzer.compute_position_probabilities("", 2, 2, n_synth=20, seed=0)
    assert (0, 0) in probs
    cell = probs[(0, 0)]
    assert abs(sum(cell.values()) - 1.0) < 1e-6
    assert len(cell) == 4


def test_predict_with_sample_prior():
    grid = [[-1, 2], [3, 4]]
    res = analyzer.predict_scratch_card(
        grid,
        target_num=1,
        iterations=4,
        global_iter=2,
        focus_iter=2,
        history_dir="",
        sample_gamma=1.0,
    )
    assert "predictions" in res

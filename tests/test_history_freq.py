import analyzer


def test_compute_history_frequency():
    freq = analyzer.compute_history_frequency("", 2, 2, 2, n_synth=20, seed=0)
    assert freq.shape == (2, 2)
    assert freq.sum() == 20


def test_predict_with_history():
    grid = [[-1, -1], [-1, -1]]
    result = analyzer.predict_scratch_card(
        grid,
        target_num=2,
        iterations=4,
        global_iter=2,
        focus_iter=2,
        history_dir="",
        gamma_history=1.0,
    )
    assert "predictions" in result

import numpy as np

import analyzer


def test_pure_sample_branch(tmp_path):
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    samples = tmp_path / "samples"
    samples.mkdir()
    boards = [
        [[1, 2], [3, 4]],
        [[1, 3], [2, 4]],
    ]
    np.savez(samples / "2x2.npz", boards=np.array(boards, dtype=np.int8))
    counts = np.zeros((2, 2, 5), dtype=float)
    for board in boards:
        for r in range(2):
            for c in range(2):
                counts[r, c, board[r][c]] += 1
    totals = counts.sum(axis=2, keepdims=True)
    totals[totals == 0] = 1
    freq = counts / totals
    out_npz = tmp_path / "out_npz"
    out_npz.mkdir()
    np.savez(out_npz / "global_pos_freq_2x2.npz", freq=freq)
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer.compute_position_probabilities.cache_clear()
    analyzer.load_global_pos_freq_npz((2, 2), out_npz)

    grid = [[1, 2], [3, -1]]
    res = analyzer.predict_scratch_card(grid, target_num=4, history_dir=str(samples))
    assert res["mode"] == "sample_only"
    assert res["strategy"] == "pure_sample+global"
    assert res["predictions"][0]["row"] == 1
    assert res["predictions"][0]["col"] == 1
    assert abs(res["predictions"][0]["score"] - 1.3) < 1e-6


def test_pure_sample_neighbor_ranking(tmp_path):
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    samples = tmp_path / "samples"
    samples.mkdir()
    arr = [[[1, 2, 2], [3, 4, 4], [5, 6, 7]]]
    np.savez(samples / "3x3.npz", boards=np.array(arr, dtype=np.int8))
    counts = np.zeros((3, 3, 10), dtype=float)
    for board in arr:
        for r in range(3):
            for c in range(3):
                counts[r, c, board[r][c]] += 1
    totals = counts.sum(axis=2, keepdims=True)
    totals[totals == 0] = 1
    freq = counts / totals
    out_npz = tmp_path / "out_npz"
    out_npz.mkdir(exist_ok=True)
    np.savez(out_npz / "global_pos_freq_3x3.npz", freq=freq)
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer.compute_position_probabilities.cache_clear()
    analyzer.load_global_pos_freq_npz((3, 3), out_npz)

    grid = [[1, -1, -1], [3, 4, -1], [5, 6, 7]]
    res = analyzer.predict_scratch_card(
        grid, target_num=2, history_dir=str(samples), top_n=2
    )
    preds = res["predictions"]
    assert preds[0]["row"] == 0 and preds[0]["col"] == 1
    assert preds[1]["row"] == 0 and preds[1]["col"] == 2


def test_pure_sample_final_score_weighting(tmp_path):
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    samples = tmp_path / "samples"
    samples.mkdir()
    arr = [[[1, 2], [3, 4]]]
    np.savez(samples / "2x2.npz", boards=np.array(arr, dtype=np.int8))
    counts = np.zeros((2, 2, 5), dtype=float)
    for board in arr:
        for r in range(2):
            for c in range(2):
                counts[r, c, board[r][c]] += 1
    totals = counts.sum(axis=2, keepdims=True)
    totals[totals == 0] = 1
    freq = counts / totals
    out_npz = tmp_path / "out_npz"
    out_npz.mkdir(exist_ok=True)
    np.savez(out_npz / "global_pos_freq_2x2.npz", freq=freq)
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer.compute_position_probabilities.cache_clear()
    analyzer.load_global_pos_freq_npz((2, 2), out_npz)

    grid = [[1, -1], [3, -1]]
    res = analyzer.predict_scratch_card(
        grid, target_num=4, history_dir=str(samples), top_n=2
    )
    preds = res["predictions"]
    assert preds[0]["row"] == 1 and preds[0]["col"] == 1
    assert preds[1]["row"] == 0 and preds[1]["col"] == 1
    assert abs(preds[0]["score"] - 1.3) < 1e-6
    assert abs(preds[1]["score"] - 1.1) < 1e-6


def test_neighbor_relaxed_matching(tmp_path):
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    samples = tmp_path / "samples"
    samples.mkdir()
    arr = [[[9, 2, 3], [4, 5, 6], [7, 8, 1]]]
    np.savez(samples / "3x3.npz", boards=np.array(arr, dtype=np.int8))
    counts = np.zeros((3, 3, 10), dtype=float)
    for board in arr:
        for r in range(3):
            for c in range(3):
                counts[r, c, board[r][c]] += 1
    totals = counts.sum(axis=2, keepdims=True)
    totals[totals == 0] = 1
    freq = counts / totals
    out_npz = tmp_path / "out_npz"
    out_npz.mkdir(exist_ok=True)
    np.savez(out_npz / "global_pos_freq_3x3.npz", freq=freq)
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer.compute_position_probabilities.cache_clear()
    analyzer.load_global_pos_freq_npz((3, 3), out_npz)

    grid = [[1, 2, -1], [4, 5, -1], [7, 8, -1]]
    res = analyzer.predict_scratch_card(grid, target_num=3, history_dir=str(samples))
    assert res["mode"] == "sample_only"
    assert res["strategy"] == "pure_sample+global"
    pred = res["predictions"][0]
    assert pred["row"] == 1 and pred["col"] == 2

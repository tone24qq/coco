import numpy as np

import analyzer


def test_compute_position_probabilities(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    boards = np.array(
        [
            [[1, 2], [3, 4]],
            [[2, 1], [4, 3]],
        ],
        dtype=np.int8,
    )
    np.savez(samples / "2x2.npz", boards=boards)

    probs = analyzer.compute_position_probabilities(str(samples), 2, 2)
    assert (0, 0) in probs
    cell = probs[(0, 0)]
    assert abs(sum(cell.values()) - 1.0) < 1e-6
    assert cell.get(1, 0) == cell.get(2, 0)


def test_predict_with_sample_prior(tmp_path):
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
    out_npz.mkdir()
    np.savez(out_npz / "global_pos_freq_2x2.npz", freq=freq)
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer.load_all_global_pos_freqs(str(out_npz))

    grid = [[-1, 2], [3, 4]]
    res = analyzer.predict_scratch_card(
        grid,
        target_num=1,
        iterations=4,
        global_iter=2,
        focus_iter=2,
        history_dir=str(samples),
        sample_gamma=1.0,
    )
    assert "predictions" in res

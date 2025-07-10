import numpy as np

import analyzer


def test_compute_history_frequency(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    boards = np.array(
        [
            [[1, 2], [3, 4]],
            [[2, 2], [1, 2]],
        ],
        dtype=np.int8,
    )
    np.savez(samples / "2x2.npz", boards=boards)

    freq = analyzer.compute_history_frequency(str(samples), 2, 2, 2)
    assert freq.shape == (2, 2)
    assert abs(freq[0, 0] - 0.25) < 1e-6
    assert abs(freq[0, 1] - 0.5) < 1e-6
    assert abs(freq[1, 1] - 0.25) < 1e-6


def test_compute_history_frequency_precomputed(tmp_path, monkeypatch):
    prior_dir = tmp_path / "priors"
    prior_dir.mkdir()
    arr = np.array([[0.1, 0.9], [0.0, 0.0]])
    np.save(prior_dir / "2x2.npy", arr)
    monkeypatch.setattr(analyzer, "PRIORS_DIR", prior_dir)
    analyzer._PRIOR_CACHE.clear()
    freq = analyzer.compute_history_frequency(str(tmp_path / "samples"), 2, 2, 2)
    assert np.allclose(freq, arr)


def test_predict_with_history(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    arr = [[[2, 1], [3, 2]]]
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

    grid = [[-1, -1], [-1, -1]]
    result = analyzer.predict_scratch_card(
        grid,
        target_num=2,
        iterations=4,
        global_iter=2,
        focus_iter=2,
        history_dir=str(samples),
        gamma_history=1.0,
    )
    assert "predictions" in result

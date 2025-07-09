import numpy as np

import analyzer


def test_heatmap_fallback_when_sparse(tmp_path):
    rows, cols = 8, 10
    out_npz = tmp_path / "out_npz"
    out_npz.mkdir()
    freq = np.zeros((rows, cols, rows * cols + 1), dtype=float)
    freq[0, 0, 1] = 1.0
    totals = freq.sum(axis=2, keepdims=True)
    totals[totals == 0] = 1
    freq /= totals
    np.savez(out_npz / f"global_pos_freq_{rows}x{cols}.npz", freq=freq)
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer.load_all_global_pos_freqs(str(out_npz))

    samples = tmp_path / "samples"
    samples.mkdir()

    grid = [[-1 for _ in range(cols)] for _ in range(rows)]
    positions = [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (4, 0),
        (4, 1),
    ]
    for idx, (r, c) in enumerate(positions, start=2):
        grid[r][c] = idx
    grid[0][0] = -1

    res = analyzer.predict_scratch_card(
        grid,
        target_num=1,
        history_dir=str(samples),
        sample_gamma=0.5,
        result_top_k=1,
    )
    assert res["strategy"] == "heatmap_global_only"
    assert res["fallback_heat"] is True
    pred = res["predictions"][0]
    assert pred["row"] == 0 and pred["col"] == 0

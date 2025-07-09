import numpy as np

import analyzer


def test_compute_global_distribution_fallback(tmp_path, monkeypatch):
    rows, cols = 2, 2
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    freq = np.ones((rows, cols, rows * cols + 1))
    totals = freq.sum(axis=2, keepdims=True)
    freq = freq / totals
    np.savez(out_dir / f"global_pos_freq_{rows}x{cols}.npz", freq=freq)
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    monkeypatch.setattr(analyzer, "ENABLE_ZIP_JSON", False)
    monkeypatch.setattr(
        analyzer,
        "load_global_pos_freq_npz",
        lambda shape, npz_dir=out_dir: analyzer._load_global_pos_freq_npz_cached(
            shape, out_dir
        ),
    )
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer._NPZ_CACHE.clear()
    res = analyzer.compute_global_distribution(str(samples_dir), rows, cols)
    assert np.allclose(res, freq)

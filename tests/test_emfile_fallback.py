import numpy as np

import analyzer


def test_emfile_retry(monkeypatch, tmp_path):
    rows, cols = 2, 2
    out_dir = tmp_path
    arr = np.ones((rows, cols, rows * cols + 1))
    np.savez(out_dir / "global_pos_freq_2x2.npz", freq=arr)
    call = {"n": 0}

    orig_load = np.load

    def fake_load(*args, **kwargs):
        call["n"] += 1
        if call["n"] == 1:
            raise OSError(24, "Too many open files")
        return orig_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", fake_load)
    analyzer._NPZ_CACHE.clear()
    loaded = analyzer.load_global_pos_freq_npz((2, 2), out_dir)
    assert np.array_equal(loaded, arr)
    assert call["n"] >= 2

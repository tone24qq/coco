import numpy as np

import analyzer


def test_predict_fallback_to_compute(monkeypatch):
    called = {"n": 0}

    def fake_load(shape, npz_dir=analyzer.DEFAULT_NPZ_DIR):
        raise FileNotFoundError

    def fake_compute(samples_dir, rows, cols):
        called["n"] += 1
        return np.zeros((rows, cols, rows * cols + 1))

    monkeypatch.setattr(analyzer, "load_global_pos_freq_npz", fake_load)
    monkeypatch.setattr(analyzer, "compute_global_distribution", fake_compute)
    monkeypatch.setattr(analyzer, "load_samples_for_shape", lambda *_: [])
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    monkeypatch.setattr(analyzer, "get_global_pos_freq", lambda *_: None)

    grid = [[-1, -1], [-1, -1]]
    analyzer.predict_scratch_card(grid, target_num=1, use_neighbor_lock=False)
    assert called["n"] == 2

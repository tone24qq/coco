import numpy as np

import analyzer


def test_npz_metrics(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    arr = np.ones((2, 2, 5))
    np.savez(out_dir / "global_pos_freq_2x2.npz", freq=arr)
    analyzer._NPZ_CACHE.clear()
    before = analyzer.NPZ_LOADED_TOTAL.labels("2x2")._value.get()
    analyzer.load_global_pos_freq_npz((2, 2), out_dir)
    after = analyzer.NPZ_LOADED_TOTAL.labels("2x2")._value.get()
    assert after - before == 1
    assert analyzer.NPZ_CACHE_BYTES._value.get() > 0
    assert analyzer.NPZ_LOAD_LATENCY_SECONDS.labels("2x2")._value.get() >= 0

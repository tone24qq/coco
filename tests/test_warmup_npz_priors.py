import asyncio

import numpy as np

import analyzer
import app
import brain


def test_warm_up_generates_priors_from_npz(tmp_path, monkeypatch):
    out_npz = tmp_path / "out"
    out_npz.mkdir()
    freq = np.zeros((2, 2, 5), dtype=float)
    freq[0, 0, 1] = 1.0
    freq[0, 1, 2] = 1.0
    freq[1, 0, 3] = 1.0
    freq[1, 1, 4] = 1.0
    np.savez(out_npz / "global_pos_freq_2x2.npz", freq=freq)

    priors_dir = tmp_path / "priors"
    priors_dir.mkdir()

    monkeypatch.setattr(app, "PRIORS_DIR", priors_dir)
    monkeypatch.setattr(analyzer, "DEFAULT_NPZ_DIR", out_npz)
    monkeypatch.setattr(app, "_load_samples_background", lambda: None)

    brain.priors_map.clear()
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    asyncio.run(app.warm_up())

    assert "2x2" in brain.priors_map
    assert brain.priors_map["2x2"][(0, 0)][1] == 1.0

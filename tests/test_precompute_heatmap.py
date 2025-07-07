import json
import zipfile

import numpy as np

import analyzer


def _make_samples(path):
    data = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    with zipfile.ZipFile(path / "s.zip", "w") as zf:
        zf.writestr("a.json", json.dumps(data))


def test_precompute_npz(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    _make_samples(samples)
    freq = analyzer.compute_global_distribution(str(samples), 2, 2)
    out = samples / "pos_freq_2x2.npz"
    np.savez(out, freq=freq)
    analyzer.compute_global_distribution.cache_clear()
    loaded = analyzer.compute_global_distribution(str(samples), 2, 2)
    assert np.allclose(freq, loaded)

import json
import zipfile

import numpy as np

import precompute_heatmap


def test_precompute_heatmap(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    data = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("a.json", json.dumps(data))

    monkeypatch.chdir(tmp_path)
    precompute_heatmap.main()

    out = samples / "pos_freq_2x2.npz"
    assert out.exists()
    freq = np.load(out)["freq"]
    assert freq.shape == (2, 2, 5)

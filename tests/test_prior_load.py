import numpy as np

import analyzer


def test_prior_load(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    grid = [[1, 2], [3, 4]]
    np.savez(samples / "2x2.npz", boards=np.array([grid], dtype=np.int8))
    cube = np.zeros((2, 2, 5), dtype=np.int64)
    arr = np.array(grid)
    rr, cc = np.indices(arr.shape)
    mask = arr >= 1
    np.add.at(cube, (rr[mask], cc[mask], arr[mask]), 1)
    np.save(samples / "prior.npy", cube)
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer._NPZ_CACHE.clear()
    monkeypatch.setattr(
        analyzer,
        "load_global_pos_freq_npz",
        lambda *_: (_ for _ in ()).throw(FileNotFoundError()),
    )
    probs = analyzer.compute_position_probabilities(str(samples), 2, 2)
    assert probs[(0, 0)][1] == 1.0

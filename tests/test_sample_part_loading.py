import numpy as np

import analyzer


def test_load_samples_from_parts(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    arr = np.array([[[1, 2], [3, 4]]], dtype=np.int8)
    np.savez(samples / "boards_2x2_part0.npz", boards=arr)

    res = analyzer.load_samples_for_shape(str(samples), 2, 2)
    assert len(res) == 1
    board, name = res[0]
    assert name.startswith("boards_2x2_part0.npz")
    assert np.array_equal(board, arr[0])

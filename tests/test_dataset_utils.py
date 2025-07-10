import numpy as np

from dataset_utils import load_boards_from_npz


def test_load_boards_from_npz(tmp_path):
    npz_path = tmp_path / "2x2.npz"
    arr = np.array(
        [
            [[1, 2], [3, 4]],
            [[4, 3], [2, 1]],
        ],
        dtype=np.uint8,
    )
    np.savez(npz_path, boards=arr)

    result = load_boards_from_npz(npz_path)
    assert np.array_equal(np.array(result), arr)

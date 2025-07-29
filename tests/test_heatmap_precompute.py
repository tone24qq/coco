import numpy as np

from dataset import BLANK_VALUE
from precompute_heatmap import _update_stats, bucket_of, save_statistics


def test_update_stats_counts_blank_only() -> None:
    board = np.array([[1, BLANK_VALUE], [BLANK_VALUE, 2]])
    heatmaps = {}
    counts = {}
    _update_stats(board, target=2, heatmaps=heatmaps, counts=counts)
    shape = (2, 2, bucket_of(2))
    expected_hits = np.array([[0.0, 0.0], [0.0, 1.0]])
    expected_counts = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(heatmaps[shape], expected_hits)
    np.testing.assert_array_equal(counts[shape], expected_counts)


def test_save_statistics_bucket_and_blank_only(tmp_path) -> None:
    board = np.array([[1, BLANK_VALUE], [BLANK_VALUE, 2]])
    heatmaps = {}
    counts = {}
    _update_stats(board, target=2, heatmaps=heatmaps, counts=counts)
    save_statistics(heatmaps, counts, str(tmp_path))
    fname = tmp_path / "heatmap_small_2x2.npy"
    assert fname.exists()
    arr = np.load(fname)
    assert arr.shape == (2, 2)
    # value at target position should be highest
    assert arr[1, 1] > arr[0, 1]

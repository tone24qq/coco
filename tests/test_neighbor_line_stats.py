import numpy as np
import pytest

import analyzer
from neighbor_line_stats import compute_neighbor_line_stats


def test_score_shape_and_range(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    boards = np.array(
        [
            [[1, 2], [3, 4]],
            [[1, 3], [2, 4]],
        ],
        dtype=np.int16,
    )
    np.savez(samples / "boards_2x2_part1.npz", boards=boards)

    grid = np.array([[1, -1], [3, -1]])
    score = compute_neighbor_line_stats(
        grid,
        target_num=4,
        samples_dir=str(samples),
    )
    assert score.shape == grid.shape
    assert np.isfinite(score).all()
    assert 0.0 <= float(score.max()) <= 1.0
    assert score[0, 1] == pytest.approx(1.0)
    assert score[1, 1] == pytest.approx(1.0)


def test_invalid_mode():
    grid = np.array([[1, -1], [3, -1]])
    with pytest.raises(ValueError):
        compute_neighbor_line_stats(
            grid,
            target_num=4,
            enable_neighbor_match=False,
            enable_line_match=False,
        )


def test_sample_neighbor_line_stats(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    boards = np.array(
        [
            [[1, 4], [3, 2]],
            [[1, 2], [3, 4]],
        ],
        dtype=np.int16,
    )
    np.savez(samples / "boards_2x2_part1.npz", boards=boards)

    grid = np.array([[1, -1], [3, -1]])
    preds = analyzer.sample_neighbor_line_stats(
        grid,
        target_num=4,
        samples_dir=str(samples),
        top_k=1,
    )
    assert preds == [((0, 1), [4], [1.0])]

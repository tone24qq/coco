import numpy as np
import pytest

import analyzer
from neighbor_line_stats import compute_neighbor_line_stats
from strategy_types import Strategy


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


def test_predict_strategy_sample_line(tmp_path):
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
    freq = np.zeros((2, 2, 5), dtype=int)
    for b in boards:
        for r in range(2):
            for c in range(2):
                freq[r, c, b[r, c]] += 1
    meta = np.frombuffer(b"{}", dtype=np.uint8)
    np.savez(samples / "sample_stats_2x2.npz", freq=freq, meta=meta)

    grid = [[1, -1], [3, -1]]
    res = analyzer.predict_scratch_card(
        grid,
        target_num=4,
        history_dir=str(samples),
        strategy=Strategy.SAMPLE_LINE,
        top_n=1,
    )
    pred = res["predictions"][0]
    assert res["strategy"] == "sample_line"
    assert pred["row"] == 0 and pred["col"] == 1

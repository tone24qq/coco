import numpy as np

from modules import (compute_difference_trend, compute_focus_score,
                     connectivity_heatmap, detect_mirror_sequences,
                     detect_skip_patterns, fuse_scores, sequence_tail_analyzer)


def _sample_grid():
    grid = np.array(
        [
            [1, -1, 3, 4],
            [5, 6, -1, 8],
            [9, 10, 11, 12],
            [13, 14, 15, -1],
        ]
    )
    return grid


def test_compute_focus_score_shape():
    grid = _sample_grid()
    out = compute_focus_score(grid)
    assert out.shape == grid.shape
    assert np.isfinite(out).all()


def test_detect_skip_patterns_shape():
    grid = _sample_grid()
    out = detect_skip_patterns(grid)
    assert out.shape == grid.shape


def test_difference_trend_shape():
    grid = _sample_grid()
    out = compute_difference_trend(grid)
    assert out.shape == grid.shape


def test_mirror_sequences_shape():
    grid = _sample_grid()
    out = detect_mirror_sequences(grid)
    assert out.shape == grid.shape


def test_connectivity_heatmap_shape():
    grid = _sample_grid()
    out = connectivity_heatmap(grid)
    assert out.shape == grid.shape


def test_sequence_tail_analyzer_shape():
    grid = _sample_grid()
    out = sequence_tail_analyzer(grid)
    assert out.shape == grid.shape


def test_fuse_scores_basic():
    grid = _sample_grid()
    scores = {
        "focus": compute_focus_score(grid),
        "skip": detect_skip_patterns(grid),
        "diff": compute_difference_trend(grid),
        "mirror": detect_mirror_sequences(grid),
        "conn": connectivity_heatmap(grid),
        "tail": sequence_tail_analyzer(grid),
    }
    fused = fuse_scores(scores, grid)
    assert fused.shape == grid.shape
    assert np.isfinite(fused).all()

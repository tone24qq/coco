# isort: skip_file
import numpy as np

from modules import (
    compute_difference_trend,
    compute_focus_score,
    connectivity_heatmap,
    detect_mirror_sequences,
    detect_skip_patterns,
    fuse_scores,
    sequence_tail_analyzer,
)


def _sample_grid() -> np.ndarray:
    return np.array(
        [
            [1, -1, 3],
            [-1, 5, 6],
            [7, 8, -1],
        ]
    )


def test_individual_module_shapes():
    grid = _sample_grid()
    fns = [
        compute_focus_score,
        detect_skip_patterns,
        compute_difference_trend,
        detect_mirror_sequences,
        connectivity_heatmap,
        sequence_tail_analyzer,
    ]
    for fn in fns:
        s = fn(grid)
        assert s.shape == grid.shape
        assert np.all(np.isfinite(s))


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
    assert np.all(np.isfinite(fused))
    assert np.all(fused[grid != -1] == 0)

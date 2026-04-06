from pathlib import Path

import numpy as np

from src.masking_eval.candidate_scoring import legal_candidates, score_candidate
from src.masking_eval.data_loader import load_full_boards


def test_board_audit_passes_sample() -> None:
    boards, audit = load_full_boards(Path("samples/data/full_boards_10x8.json"))
    assert audit.valid_boards > 0
    assert audit.invalid_boards == 0
    assert boards[0].grid.shape == (10, 8)


def test_candidate_scoring_uses_masked_grid_only_shape() -> None:
    grid = np.arange(1, 81).reshape(10, 8)
    masked = grid.copy()
    masked[0, 0] = -1
    cands = legal_candidates(masked)
    assert grid[0, 0] in cands
    feat = score_candidate(
        masked,
        (0, 0),
        cands[0],
        heatmap_prior=np.zeros((10, 8)),
        modules=["focus", "skip"],
    )
    assert set(feat.keys()) == {"focus", "skip"}

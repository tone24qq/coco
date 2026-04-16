from __future__ import annotations

from src.masking_dataset import MaskingConfig, build_masked_ranking_dataset
from src.whole_board_features import compute_board_state_features, compute_candidate_delta_features


def test_whole_board_features_have_expected_keys() -> None:
    board = [[1, -1], [3, 4]]
    state = compute_board_state_features(board, target_number=2)
    delta = compute_candidate_delta_features(board, target_number=2, cand_row=0, cand_col=1, board_state_features=state)
    assert "known_ratio" in state
    assert "delta_known_ratio" in delta
    assert delta["is_feasible"] == 1.0


def test_masking_dataset_lineage_stable() -> None:
    boards = [
        {
            "board_id": "b1",
            "rows": 2,
            "cols": 2,
            "size_class": "2x2",
            "grid": [[1, 2], [3, 4]],
            "source_type": "real",
        }
    ]
    df = build_masked_ranking_dataset(boards, MaskingConfig(ratios=[0.5], masks_per_ratio=1))
    assert not df.empty
    assert set(df["lineage_id"].unique()) == {"b1"}
    assert (df.groupby("group_id")["label"].sum() == 1).all()

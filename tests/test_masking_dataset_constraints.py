from __future__ import annotations

from src.masking_dataset import MaskingConfig, build_masked_ranking_dataset


def test_groups_are_truly_masked_and_single_positive() -> None:
    boards = [
        {
            "board_id": "b1",
            "lineage_id": "b1",
            "rows": 2,
            "cols": 2,
            "size_class": "2x2",
            "source_type": "real",
            "grid": [[1, 2], [3, 4]],
        }
    ]
    df = build_masked_ranking_dataset(boards, MaskingConfig(ratios=[0.5], masks_per_ratio=2))
    assert not df.empty
    assert (df["is_feasible"] == 1).all()
    g = df.groupby("group_id")
    assert (g["label"].sum() == 1).all()
    assert (g.size() >= 2).all()

from __future__ import annotations

import pandas as pd

from src.masking_dataset import (
    MaskingConfig,
    build_masked_ranking_dataset,
    iter_masked_ranking_dataset_chunks,
    iter_masked_ranking_rows,
)


def _boards() -> list[dict[str, object]]:
    return [
        {
            "board_id": "b1",
            "lineage_id": "b1",
            "rows": 2,
            "cols": 2,
            "size_class": "2x2",
            "source_type": "real",
            "grid": [[1, 2], [3, 4]],
            "source_file": "x.xlsx",
            "sheet_name": "S1",
        }
    ]


def test_streaming_chunks_keep_contract() -> None:
    cfg = MaskingConfig(ratios=[0.5], masks_per_ratio=2)
    chunks = list(iter_masked_ranking_dataset_chunks(_boards(), cfg, chunk_rows=5))
    assert chunks
    total = sum(len(c) for c in chunks)
    assert total > 0
    assert all(len(c) <= 5 for c in chunks[:-1])
    merged = pd.concat(chunks, ignore_index=True)
    assert "group_id" in merged.columns
    assert "label" in merged.columns
    assert "size_class" in merged.columns


def test_streaming_is_deterministic() -> None:
    cfg = MaskingConfig(ratios=[0.5], masks_per_ratio=1)
    run1 = [r["group_id"] for r in iter_masked_ranking_rows(_boards(), cfg)]
    run2 = [r["group_id"] for r in iter_masked_ranking_rows(_boards(), cfg)]
    assert run1 == run2


def test_streaming_matches_legacy_semantics() -> None:
    cfg = MaskingConfig(ratios=[0.5], masks_per_ratio=2)
    legacy = (
        build_masked_ranking_dataset(_boards(), cfg)
        .sort_values(["group_id", "cand_row", "cand_col"])
        .reset_index(drop=True)
    )
    streamed = pd.concat(list(iter_masked_ranking_dataset_chunks(_boards(), cfg, chunk_rows=3)), ignore_index=True)
    streamed = streamed.sort_values(["group_id", "cand_row", "cand_col"]).reset_index(drop=True)
    assert len(legacy) == len(streamed)
    assert legacy["group_id"].tolist() == streamed["group_id"].tolist()
    assert legacy["label"].sum() == streamed["label"].sum()

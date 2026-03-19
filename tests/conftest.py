from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.build_features import build_feature_rows, write_feature_store
from src.ranking_dataset import attach_group_ids, write_rows
from src.utils import DrawRecord


@pytest.fixture
def synthetic_records() -> list[DrawRecord]:
    records = []
    for i in range(180):
        start = (i % 60) + 1
        nums = tuple(sorted(((start + k - 1) % 80) + 1 for k in range(20)))
        records.append(
            DrawRecord(
                issue=f"{20260101000 + i}",
                draw_date=date(2026, 1, 1) + timedelta(days=i // 30),
                numbers=nums,
                day_issue_index=(i % 30) + 1,
            )
        )
    return records


@pytest.fixture
def ranking_dataset_path(tmp_path: Path, synthetic_records: list[DrawRecord]) -> Path:
    features = build_feature_rows(synthetic_records, min_history=100, retrieval_window=40, top_k=8)
    feature_path = tmp_path / "ranking_features.csv"
    write_feature_store(feature_path, features)

    rows = pd.read_csv(feature_path).to_dict(orient="records")
    rows_with_group = attach_group_ids(rows)
    dataset_path = tmp_path / "ranking_dataset.csv"
    write_rows(dataset_path, rows_with_group)
    return dataset_path

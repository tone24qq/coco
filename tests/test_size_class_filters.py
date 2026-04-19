from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from scripts.split_ranking_dataset import split_df


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_generate_synthetic_size_class_filter(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    profile = tmp_path / "profile.json"
    out = tmp_path / "synth.jsonl"
    _write_jsonl(
        real,
        [
            {
                "board_id": "b1",
                "rows": 4,
                "cols": 5,
                "size_class": "4x5",
                "grid": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
            },
            {
                "board_id": "b2",
                "rows": 6,
                "cols": 10,
                "size_class": "6x10",
                "grid": [list(range(i * 10 + 1, i * 10 + 11)) for i in range(6)],
            },
        ],
    )
    profile.write_text(
        json.dumps(
            {
                "4x5": {
                    "rows": 4,
                    "cols": 5,
                    "size_class": "4x5",
                    "feature_means": {
                        "tail_entropy": 0.0,
                        "same_tail_adjacency_rate": 0.0,
                        "same_decade_proximity_rate": 0.0,
                        "consecutive_neighbor_rate": 0.0,
                        "row_known_entropy": 0.0,
                        "col_known_entropy": 0.0,
                        "edge_center_balance": 0.0,
                    },
                    "feature_stds": {
                        "tail_entropy": 1.0,
                        "same_tail_adjacency_rate": 1.0,
                        "same_decade_proximity_rate": 1.0,
                        "consecutive_neighbor_rate": 1.0,
                        "row_known_entropy": 1.0,
                        "col_known_entropy": 1.0,
                        "edge_center_balance": 1.0,
                    },
                },
                "6x10": {
                    "rows": 6,
                    "cols": 10,
                    "size_class": "6x10",
                    "feature_means": {
                        "tail_entropy": 0.0,
                        "same_tail_adjacency_rate": 0.0,
                        "same_decade_proximity_rate": 0.0,
                        "consecutive_neighbor_rate": 0.0,
                        "row_known_entropy": 0.0,
                        "col_known_entropy": 0.0,
                        "edge_center_balance": 0.0,
                    },
                    "feature_stds": {
                        "tail_entropy": 1.0,
                        "same_tail_adjacency_rate": 1.0,
                        "same_decade_proximity_rate": 1.0,
                        "consecutive_neighbor_rate": 1.0,
                        "row_known_entropy": 1.0,
                        "col_known_entropy": 1.0,
                        "edge_center_balance": 1.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    res = subprocess.run(
        [
            "python",
            "scripts/generate_synthetic_boards.py",
            "--size-class",
            "4x5",
            "--real-corpus",
            str(real),
            "--profile",
            str(profile),
            "--output",
            str(out),
            "--per-real",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    payload = json.loads(res.stdout.strip())
    assert payload["selected_size_class"] == "4x5"
    assert payload["real_board_count_after_filter"] == 1


def test_build_masked_size_class_empty_failfast(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    _write_jsonl(
        real,
        [
            {
                "board_id": "b1",
                "rows": 4,
                "cols": 5,
                "size_class": "4x5",
                "grid": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                "source_type": "real",
            }
        ],
    )
    synth = tmp_path / "synth.jsonl"
    synth.write_text("", encoding="utf-8")
    out = tmp_path / "rank.parquet"
    res = subprocess.run(
        [
            "python",
            "scripts/build_masked_ranking_dataset.py",
            "--size-class",
            "10x10",
            "--real-corpus",
            str(real),
            "--synthetic-corpus",
            str(synth),
            "--output",
            str(out),
            "--mask-ratios",
            "0.5",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    assert res.returncode != 0
    assert "no boards after size_class filter" in (res.stderr + res.stdout)


def test_split_fallback_prefers_real_key() -> None:
    df = pd.DataFrame(
        [
            {"board_id": "s1", "lineage_id": "s1", "source_type": "synthetic", "group_id": "g1"},
            {"board_id": "r1", "lineage_id": "r1", "source_type": "real", "group_id": "g2"},
        ]
    )
    out = split_df(
        df,
        holdout_ratio=0.0,
        split_mode="by_board",
        seed=42,
        include_synth_in_holdout=True,
        valid_real_only=False,
        holdout_real_only=False,
        exclude_synth_from_valid=False,
    )
    holdout_keys = set(out["holdout"]["board_id"].tolist())
    assert "r1" in holdout_keys

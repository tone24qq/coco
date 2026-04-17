from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def _make_dataset(path: Path, size_class: str = "2x2") -> None:
    rows = []
    for gid in range(6):
        for cand in range(3):
            rows.append(
                {
                    "group_id": f"g{gid}",
                    "lineage_id": f"l{gid}",
                    "board_id": f"b{gid}",
                    "source_type": "real",
                    "size_class": size_class,
                    "cand_row": cand + 1,
                    "cand_col": 1,
                    "label": 1 if cand == 0 else 0,
                    "is_feasible": 1,
                    "board_state_a": float(cand),
                    "candidate_delta_a": float(3 - cand),
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_tuning_pipeline_and_retrain(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    valid = tmp_path / "valid.parquet"
    holdout = tmp_path / "holdout.parquet"
    for p in (train, valid, holdout):
        _make_dataset(p)

    art = tmp_path / "artifacts"
    rep = tmp_path / "reports"

    subprocess.run(
        [
            "python",
            "scripts/tune_local_ranker.py",
            "--train-path",
            str(train),
            "--valid-path",
            str(valid),
            "--holdout-path",
            str(holdout),
            "--backend",
            "sklearn",
            "--n-trials",
            "2",
            "--search-method",
            "random",
            "--artifacts-dir",
            str(art),
            "--report-dir",
            str(rep),
        ],
        check=True,
    )

    assert (rep / "tuning_trials.csv").exists()
    assert (rep / "tuning_summary.json").exists()
    assert (rep / "tuning_leaderboard.json").exists()
    assert (art / "best_params.json").exists()
    assert (art / "main_ranker.pkl").exists()

    summary = json.loads((rep / "tuning_summary.json").read_text(encoding="utf-8"))
    assert summary["n_trials"] >= 2
    assert "final_holdout_metrics" in summary


def test_safe_io_manifest_roundtrip(tmp_path: Path) -> None:
    from src.safe_io import SafeWriteConfig, read_dataset_auto, write_dataframe_safe

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out = tmp_path / "tiny.parquet"
    meta = write_dataframe_safe(
        df,
        out,
        fmt="parquet",
        config=SafeWriteConfig(max_file_mb=100, producer_script="test"),
        shard_rows=1,
    )
    if meta["type"] == "dataset_dir":
        loaded = read_dataset_auto(Path(meta["path"]))
    else:
        loaded = read_dataset_auto(out)
    assert len(loaded) == 3

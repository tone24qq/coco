import json
from pathlib import Path

import pandas as pd
import pytest

from src.train_transformer import train_model


def _make_history(path: Path, start: int = 1000, end: int = 1120) -> None:
    rows = []
    for issue in range(start, end):
        rows.append(
            {
                "issue": issue,
                "draw_time": "2026-01-01",
                **{f"n{i}": ((issue + i) % 80) + 1 for i in range(1, 21)},
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_max_issues_uses_recent_n(tmp_path: Path) -> None:
    input_path = tmp_path / "history.csv"
    _make_history(input_path, 1000, 1200)
    output_dir = tmp_path / "model"

    train_model(
        input_path=input_path,
        output_dir=output_dir,
        model_file="model.ckpt",
        window_size=20,
        seed=42,
        epochs=1,
        batch_size=32,
        alpha=0.2,
        stale_threshold=20,
        max_issues=60,
    )

    meta = json.loads((output_dir / "transformer_metadata.json").read_text("utf-8"))
    if meta.get("source_issue_count") != 200 or meta.get("used_issue_count") != 60:
        pytest.fail("max-issues slicing metadata mismatch")


def test_max_issues_invalid_fail_fast(tmp_path: Path) -> None:
    input_path = tmp_path / "history.csv"
    _make_history(input_path)
    with pytest.raises(ValueError, match="max-issues"):
        train_model(
            input_path=input_path,
            output_dir=tmp_path / "model",
            model_file="model.ckpt",
            window_size=20,
            seed=42,
            epochs=1,
            batch_size=32,
            alpha=0.2,
            stale_threshold=20,
            max_issues=0,
        )


def test_early_stopping_patience_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "history.csv"
    _make_history(input_path, 1000, 1060)

    calls = {"count": 0}

    def fake_evaluate(model, loader):
        calls["count"] += 1
        return 0.1, {
            "valid_hit_at_20": 0.0,
            "valid_hit_at_10": 0.0,
            "valid_hit_at_3": 0.0,
            "valid_ndcg_at_20": 0.1,
            "valid_top3_at_least_one_hit": 0.0,
        }

    monkeypatch.setattr("src.train_transformer._evaluate", fake_evaluate)

    train_model(
        input_path=input_path,
        output_dir=tmp_path / "model",
        model_file="model.ckpt",
        window_size=20,
        seed=42,
        epochs=10,
        batch_size=32,
        alpha=0.2,
        stale_threshold=20,
        max_issues=50,
    )

    if calls["count"] != 5:
        pytest.fail("early stopping patience should stop after 3 no-improve epochs")


def test_training_outputs_chinese_progress(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "history.csv"
    _make_history(input_path, 1000, 1080)

    train_model(
        input_path=input_path,
        output_dir=tmp_path / "model",
        model_file="model.ckpt",
        window_size=20,
        seed=42,
        epochs=1,
        batch_size=32,
        alpha=0.2,
        stale_threshold=20,
        max_issues=60,
    )

    out = capsys.readouterr().out
    if "[進度]" not in out or "%" not in out or "訓練完成" not in out:
        pytest.fail("training should print chinese progress with percentage")

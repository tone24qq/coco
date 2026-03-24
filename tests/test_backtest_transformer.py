from pathlib import Path

import pandas as pd
import pytest

from src.backtest_transformer import run_backtest
from src.model_transformer import SmallTransformerRanker, TransformerConfig


def _make_history(path: Path) -> None:
    rows = []
    for issue in range(1000, 1030):
        rows.append(
            {
                "issue": issue,
                "draw_time": "2026-01-01",
                **{f"n{i}": ((issue + i) % 80) + 1 for i in range(1, 21)},
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_backtest_transformer_smoke(tmp_path: Path) -> None:
    input_path = tmp_path / "history.csv"
    _make_history(input_path)

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    SmallTransformerRanker(TransformerConfig(feature_dim=24)).save(
        model_dir / "model.ckpt"
    )

    out = tmp_path / "backtest"
    run_backtest(input_path, model_dir / "model.ckpt", out, window_size=20)
    if not (out / "summary.json").exists():
        pytest.fail("backtest summary missing")

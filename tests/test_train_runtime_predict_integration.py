import json
from pathlib import Path

import pandas as pd
import pytest

from src.inference import predict
from src.runtime_history import build_runtime_history
from src.train_transformer import train_model


def test_train_runtime_predict_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history_path = tmp_path / "history.csv"
    rows = []
    for issue in range(1000, 1120):
        rows.append(
            {
                "issue": issue,
                "draw_time": "2026-01-01",
                **{f"n{i}": ((issue + i) % 80) + 1 for i in range(1, 21)},
            }
        )
    pd.DataFrame(rows).to_csv(history_path, index=False)

    model_dir = tmp_path / "models"
    train_model(
        input_path=history_path,
        output_dir=model_dir,
        model_file="model.ckpt",
        window_size=50,
        seed=42,
        epochs=2,
        batch_size=16,
        alpha=0.2,
        stale_threshold=20,
    )

    runtime_dir = tmp_path / "runtime"
    build_runtime_history(history_path, runtime_dir, model_dir)

    cfg = {
        "auto_fetch_sources": [{"name": "mock", "url": "https://mock"}],
        "fetch": {"timeout_seconds": 1.0, "retries": 0, "backoff_seconds": 0.0},
        "runtime": {
            "local_history_path": str(history_path),
            "runtime_dir": str(runtime_dir),
        },
        "model": {
            "artifact_file": "model.ckpt",
            "model_version": "small_transformer_v2",
            "feature_version": "rank_window_v2",
            "window_size": 50,
            "seed": 42,
            "stale_threshold": 20,
        },
    }
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            [{"issue": "1120", "draw_time": "x", "numbers": list(range(1, 21))}],
            "mock",
            [],
        ),
    )

    result = predict(runtime_dir)
    scores = result["scores"]
    if not isinstance(scores, list) or len(scores) != 80:
        pytest.fail("integration predict scores mismatch")

"""Training entrypoint for small transformer ranking model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.build_rank_windows import build_training_samples
from src.history_store import load_local_history
from src.model_transformer import SmallTransformerRanker, TransformerConfig

MODEL_VERSION = "small_transformer_v1"
FEATURE_VERSION = "rank_window_v1"


def train_model(
    input_path: Path,
    output_dir: Path,
    model_file: str,
    window_size: int,
    seed: int,
) -> None:
    history = load_local_history(input_path)

    x_train, y_train = build_training_samples(history, window_size=window_size)
    split_idx = int(len(x_train) * 0.8)
    if split_idx <= 0 or split_idx >= len(x_train):
        raise ValueError("Invalid time-series split; need more data")

    train_x = x_train[:split_idx]
    train_y = y_train[:split_idx]
    valid_x = x_train[split_idx:]

    config = TransformerConfig(seed=seed)
    model = SmallTransformerRanker(
        config=config, params=SmallTransformerRanker.init_params(config)
    )
    model.fit_head(train_x, train_y)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_file
    model.save(str(model_path))

    valid_pred = [model.predict_scores(sample) for sample in valid_x]
    valid_score = float(
        sum(float(pred.mean()) for pred in valid_pred) / len(valid_pred)
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_file": model_file,
        "window_size": window_size,
        "seed": seed,
        "train_samples": int(len(train_x)),
        "valid_samples": int(len(valid_x)),
        "trained_up_to_issue": str(history.iloc[-1]["issue"]),
        "baseline_metrics": {
            "valid_mean_score": valid_score,
            "objective": "ranking_binary_hit",
        },
        "required_input_schema": [
            "issue",
            "draw_time",
            *[f"n{i}" for i in range(1, 21)],
        ],
        "expected_output_schema": [
            "latest_known_issue",
            "target_issue",
            "model_version",
            "feature_version",
            "data_source",
            "scores",
            "top20",
            "top3",
        ],
    }
    (output_dir / "transformer_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train small transformer ranker")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-file", default="transformer_model.npz")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        input_path=args.input,
        output_dir=args.output,
        model_file=args.model_file,
        window_size=args.window_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

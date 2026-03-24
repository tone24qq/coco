"""Build versioned runtime artifacts from history CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.train_transformer import FEATURE_VERSION, MODEL_VERSION, train_model

ARTIFACT_VERSION = "runtime_history_v3"
SCORE_ARTIFACT_FILENAME = "scores.csv"
METADATA_FILENAME = "metadata.json"
MODEL_ARTIFACT_FILENAME = "transformer_model.npz"
MODEL_METADATA_FILENAME = "transformer_metadata.json"

REQUIRED_COLUMNS: List[str] = [
    "issue",
    "draw_time",
    *[f"n{i}" for i in range(1, 21)],
]
RAW_COLUMN_MAP: Dict[str, str] = {
    "期別": "issue",
    "開獎日期": "draw_time",
    **{f"獎號{i}": f"n{i}" for i in range(1, 21)},
}


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in REQUIRED_COLUMNS):
        return df[REQUIRED_COLUMNS].copy()

    if all(col in df.columns for col in RAW_COLUMN_MAP):
        normalized = df.rename(columns=RAW_COLUMN_MAP)
        return normalized[REQUIRED_COLUMNS].copy()

    raise ValueError("Input schema mismatch")


def _coerce_and_validate_numbers(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["issue"] = normalized["issue"].astype(str)
    for idx in range(1, 21):
        col = f"n{idx}"
        normalized[col] = pd.to_numeric(normalized[col], errors="raise").astype(int)
        if ((normalized[col] < 1) | (normalized[col] > 80)).any():
            raise ValueError(f"Column {col} contains values outside 1..80")
    return normalized


def _build_score_chain(normalized: pd.DataFrame) -> pd.DataFrame:
    number_cols = [f"n{i}" for i in range(1, 21)]
    counts = normalized[number_cols].stack().value_counts().sort_index()
    return pd.DataFrame(
        [
            {"number": number, "score": float(counts.get(number, 0))}
            for number in range(1, 81)
        ]
    )


def build_runtime_history(input_path: Path, output_dir: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)
    normalized = _normalize_schema(raw_df)
    if normalized.empty:
        raise ValueError("Input history is empty")

    normalized = _coerce_and_validate_numbers(normalized)
    normalized = normalized.sort_values(["issue"], kind="mergesort").reset_index(
        drop=True
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history_runtime.csv"
    score_path = output_dir / SCORE_ARTIFACT_FILENAME

    normalized.to_csv(history_path, index=False)
    _build_score_chain(normalized).to_csv(score_path, index=False)

    train_model(
        input_path=history_path,
        output_dir=output_dir,
        model_file=MODEL_ARTIFACT_FILENAME,
        window_size=100,
        seed=42,
    )

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "score_artifact": SCORE_ARTIFACT_FILENAME,
        "score_chain_size": 80,
        "history_rows": int(len(normalized)),
        "latest_issue": str(normalized.iloc[-1]["issue"]),
        "model_artifact": MODEL_ARTIFACT_FILENAME,
        "model_metadata": MODEL_METADATA_FILENAME,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "required_input_schema": REQUIRED_COLUMNS,
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
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build runtime history artifact")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_runtime_history(args.input, args.output)


if __name__ == "__main__":
    main()

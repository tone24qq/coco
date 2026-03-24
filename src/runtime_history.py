"""Build runtime artifacts and synchronize trained model artifacts (no retraining)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd

ARTIFACT_VERSION = "runtime_history_v5"
METADATA_FILENAME = "metadata.json"

REQUIRED_COLUMNS: List[str] = ["issue", "draw_time", *[f"n{i}" for i in range(1, 21)]]
RAW_COLUMN_MAP: Dict[str, str] = {
    "期別": "issue",
    "開獎日期": "draw_time",
    **{f"獎號{i}": f"n{i}" for i in range(1, 21)},
}


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in REQUIRED_COLUMNS):
        return df[REQUIRED_COLUMNS].copy()
    if all(col in df.columns for col in RAW_COLUMN_MAP):
        return df.rename(columns=RAW_COLUMN_MAP)[REQUIRED_COLUMNS].copy()
    raise ValueError("Input schema mismatch")


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["issue"] = normalized["issue"].astype(str)
    if normalized["issue"].duplicated().any():
        raise ValueError("Duplicated issue in input history")
    for idx in range(1, 21):
        col = f"n{idx}"
        normalized[col] = pd.to_numeric(normalized[col], errors="raise").astype(int)
        if ((normalized[col] < 1) | (normalized[col] > 80)).any():
            raise ValueError(f"Column {col} contains values outside 1..80")
    normalized = normalized.sort_values(["issue"], kind="mergesort").reset_index(
        drop=True
    )
    return normalized


def _build_score_chain(history: pd.DataFrame) -> pd.DataFrame:
    counts = (
        history[[f"n{i}" for i in range(1, 21)]].stack().value_counts().sort_index()
    )
    return pd.DataFrame(
        [{"number": n, "score": float(counts.get(n, 0.0))} for n in range(1, 81)]
    )


def build_runtime_history(
    input_path: Path, output_dir: Path, model_source: Path
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path)
    history = _coerce(_normalize_schema(raw))

    model_ckpt = model_source / "model.ckpt"
    model_meta = model_source / "transformer_metadata.json"
    if not model_ckpt.exists() or not model_meta.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Run training first: "
            f"expected {model_ckpt} and {model_meta}"
        )

    model_metadata = json.loads(model_meta.read_text(encoding="utf-8"))
    required_model_keys = {
        "model_version",
        "feature_version",
        "feature_names",
        "tensor_contract",
        "trained_up_to_issue",
        "baseline_metrics",
        "expected_input_schema",
        "expected_output_schema",
    }
    missing = sorted(required_model_keys - set(model_metadata.keys()))
    if missing:
        raise ValueError(f"Model metadata mismatch, missing keys: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_parquet(output_dir / "history_runtime.parquet", index=False)
    history.to_csv(output_dir / "history_runtime.csv", index=False)

    scores = _build_score_chain(history)
    scores.to_parquet(output_dir / "scores.parquet", index=False)
    scores.to_csv(output_dir / "scores.csv", index=False)

    shutil.copy2(model_ckpt, output_dir / "model.ckpt")
    shutil.copy2(model_meta, output_dir / "transformer_metadata.json")

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "storage_format": "parquet_preferred",
        "history_artifact": "history_runtime.parquet",
        "history_csv_compat": "history_runtime.csv",
        "score_artifact": "scores.parquet",
        "score_csv_compat": "scores.csv",
        "model_artifact": "model.ckpt",
        "model_metadata": "transformer_metadata.json",
        "model_version": model_metadata["model_version"],
        "feature_version": model_metadata["feature_version"],
        "trained_up_to_issue": model_metadata["trained_up_to_issue"],
        "baseline_metrics": model_metadata["baseline_metrics"],
        "feature_names": model_metadata["feature_names"],
        "tensor_contract": model_metadata["tensor_contract"],
        "stale_threshold": model_metadata.get("stale_threshold", 20),
        "expected_input_schema": model_metadata["expected_input_schema"],
        "expected_output_schema": model_metadata["expected_output_schema"],
    }
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build runtime history artifact")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model-source", type=Path, default=Path("models/transformer_v1")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_runtime_history(args.input, args.output, args.model_source)


if __name__ == "__main__":
    main()

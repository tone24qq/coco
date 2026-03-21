#!/usr/bin/env bash
set -euo pipefail

PROCESSED_PATH="${1:-data/processed/history_processed.csv}"
RUNTIME_DIR="${2:-data/runtime_history}"
OUT_DIR="${3:-dist/deploy_artifact}"

echo "[deploy] build runtime history artifact"
python -m src.runtime_history --input "$PROCESSED_PATH" --output "$RUNTIME_DIR"

echo "[deploy] package models + runtime_history"
mkdir -p "$OUT_DIR/models" "$OUT_DIR/data"
cp models/lightgbm_ranker.txt "$OUT_DIR/models/lightgbm_ranker.txt"
cp models/logistic_regression.pkl "$OUT_DIR/models/logistic_regression.pkl"
cp models/feature_columns.json "$OUT_DIR/models/feature_columns.json"
cp models/metadata.json "$OUT_DIR/models/metadata.json"
rm -rf "$OUT_DIR/data/runtime_history"
cp -R "$RUNTIME_DIR" "$OUT_DIR/data/runtime_history"

python - <<'PY'
import json
from pathlib import Path

out_dir = Path("dist/deploy_artifact")
runtime_dir = out_dir / "data" / "runtime_history"
required = [
    out_dir / "models" / "lightgbm_ranker.txt",
    out_dir / "models" / "logistic_regression.pkl",
    out_dir / "models" / "feature_columns.json",
    out_dir / "models" / "metadata.json",
    runtime_dir / "meta.json",
    runtime_dir / "numbers.npy",
    runtime_dir / "issue.npy",
    runtime_dir / "draw_date_ordinal.npy",
    runtime_dir / "day_issue_index.npy",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing deploy artifacts: {missing}")

manifest = {
    "models_dir": str((out_dir / "models").resolve()),
    "runtime_history_dir": str(runtime_dir.resolve()),
    "required_files": [str(p.relative_to(out_dir)) for p in required],
}
(out_dir / "deploy_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("[deploy] deploy_manifest.json generated")
PY

echo "[deploy] done: $OUT_DIR"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output_dir>" >&2
  exit 1
fi

OUTPUT_DIR="$1"

REQUIRED_FILES=(
  "app.py"
  "requirements.txt"
  "src/inference.py"
  "src/runtime_history.py"
  "src/__init__.py"
  "data/runtime_history/metadata.json"
  "data/runtime_history/scores.csv"
)

for required in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file for deploy bundle: $required" >&2
    exit 1
  fi
done

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/data"

cp app.py "$OUTPUT_DIR/"
cp requirements.txt "$OUTPUT_DIR/"
cp -R src "$OUTPUT_DIR/"
cp -R data/runtime_history "$OUTPUT_DIR/data/"

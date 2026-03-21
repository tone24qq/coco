#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:-deploy_bundle}"

python -m src.runtime_history --input data/processed/history_processed.csv --output data/runtime_history

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.mypy_cache' \
  --exclude '.ruff_cache' \
  ./ "$BUNDLE_DIR"/

for f in meta.json numbers.npy issue.npy draw_date_ordinal.npy day_issue_index.npy; do
  test -s "data/runtime_history/$f"
  test -s "$BUNDLE_DIR/data/runtime_history/$f"
done

echo "build_deploy_bundle.sh PASSED -> $BUNDLE_DIR"

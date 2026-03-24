# Plan

## goal
Refine the runtime transformer mainline with production-grade time-sync validation, Parquet-first storage/I/O, diversity relaxation metadata, explicit tensor/attention contracts, and artifact drift metadata while preserving the existing deploy contract.

## touched files
- PLANS.md
- README.md
- ARCHITECTURE.md
- configs/predict.yaml
- src/history_store.py
- src/runtime_history.py
- src/train_transformer.py
- src/inference.py
- tests/test_runtime_history.py
- tests/test_inference.py

## invariants
- Keep deploy commands unchanged:
  - `python -m src.runtime_history --input data/processed/history_processed.csv --output data/runtime_history`
  - `bash scripts/build_deploy_bundle.sh deploy_bundle`
- Python 3.9 / Azure Web App compatible.
- Fail-fast for missing input/artifact/schema/version/time-sync mismatch.
- Deterministic outputs with explicit tie-breakers.
- Preserve complete 1..80 score chain.
- Top3 diversity is post-ranking with explicit relaxation metadata.
- Ranking score is not calibrated probability.

## risks
- Runtime fetch source behavior can still drift over time (parser maintenance needed).
- Parquet I/O requires `pyarrow` at runtime (already in requirements).
- Time-sync fail-fast may reject responses when upstream source lags.

## validation steps
- python -m pip install -r requirements-dev.txt
- pytest -q
- flake8 .
- black --check .
- isort --check-only .
- mypy .
- bandit -r .
- pip-audit
- flake8 agent.py
- flake8 src tests
- python -m py_compile $(git ls-files '*.py')
- bash scripts/verify_mainline.sh (if file exists)
- python -m src.runtime_history --input 賓果賓果_2026.csv --output data/runtime_history
- python - <<'PY' ... TestClient('/predict') ... PY

## rollback plan
Revert the final commit to restore the previous runtime auto-fetch transformer baseline.

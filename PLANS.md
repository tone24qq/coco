# Plan

## goal
Upgrade repository mainline from numpy prototype to production PyTorch Transformer ranking pipeline with decoupled runtime artifacts, deterministic inference, walk-forward backtest, CLI predict, and strict contract/drift validations while preserving existing deploy contract.

## touched files
- PLANS.md
- requirements.txt
- README.md
- ARCHITECTURE.md
- configs/predict.yaml
- src/model_transformer.py
- src/build_rank_windows.py
- src/train_transformer.py
- src/runtime_history.py
- src/inference.py
- src/fetch_latest.py
- src/history_store.py
- src/backtest_transformer.py
- src/predict.py
- tests/test_model_transformer.py
- tests/test_build_rank_windows.py
- tests/test_history_store.py
- tests/test_train_runtime_predict_integration.py
- tests/test_inference.py
- tests/test_fetch_latest.py

## invariants
- Deploy commands unchanged:
  - `python -m src.runtime_history --input data/processed/history_processed.csv --output data/runtime_history`
  - `bash scripts/build_deploy_bundle.sh deploy_bundle`
- runtime_history does not retrain model.
- Strict time-series / walk-forward only, no random split.
- Deterministic inference and reproducible top20/top3.
- Fail-fast on artifact/schema/version/feature/tensor/drift/time-sync mismatch.
- Score semantics remain ranking_score.

## risks
- Installing torch may increase deploy image size/startup time.
- Source HTML parsers may need maintenance if upstream layouts change.
- Full training runtime may be high for large datasets; defaults tuned for local/testing.

## validation steps
- python -m pip install -r requirements-dev.txt
- python -c "import torch"
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
- python -m src.train_transformer --input 賓果賓果_2026.csv --output models/transformer_v1
- python -m src.runtime_history --input 賓果賓果_2026.csv --output data/runtime_history
- python -m src.predict --runtime-dir data/runtime_history
- python -m src.backtest_transformer --input 賓果賓果_2026.csv --output reports/transformer_backtest

## rollback plan
Revert this commit to restore previous runtime-fetch baseline.

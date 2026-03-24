# Plan

## goal
Harden the existing Transformer mainline (no model rollback) so train/backtest/CLI/API/deploy paths are contract-consistent, fail-fast, deterministic, and fully test-covered with runtime/API smoke evidence.

## touched files
- PLANS.md
- src/build_rank_windows.py
- src/history_store.py
- src/fetch_latest.py
- src/runtime_history.py
- src/inference.py
- configs/predict.yaml
- tests/test_fetch_latest.py
- tests/test_history_store.py
- tests/test_inference.py
- tests/test_app.py
- tests/test_predict_cli.py
- tests/test_backtest_transformer.py

## invariants
- Keep Transformer mainline and existing deploy contract unchanged:
  - `python -m src.runtime_history --input data/processed/history_processed.csv --output data/runtime_history`
  - `bash scripts/build_deploy_bundle.sh deploy_bundle`
- Runtime inference never retrains.
- Maintain full 1-80 score chain and deterministic top20/top3 ordering.
- Preserve time-series only behavior (no random split, no future leakage).
- Keep artifact/state_dict contracts and enforce <100MB model artifact limit.

## risks
- Source HTML layout drift can still break parsers even with stronger source-specific rules.
- Live fetch may remain unstable (upstream 503/anti-bot), affecting live smoke.
- Additional strict validations may surface existing bad local artifacts.

## validation steps
- python -m pip install -r requirements-dev.txt
- python -c "import torch"
- flake8 .
- black --check .
- isort --check-only .
- mypy .
- bandit -r .
- pip-audit
- flake8 agent.py
- flake8 src tests
- python -m py_compile $(git ls-files '*.py')
- pytest -q
- python -m src.train_transformer --input <smoke_history.csv> --output <smoke_model_dir> --epochs 2 --window-size 50
- python -m src.runtime_history --input <smoke_history.csv> --output <smoke_runtime_dir> --model-source <smoke_model_dir>
- python -m src.predict --runtime-dir <smoke_runtime_dir>
- API /predict smoke via FastAPI TestClient (single + 3 repeated calls latency + determinism)

## rollback plan
Revert the hardening commit to restore the prior transformer branch behavior.

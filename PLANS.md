# Plan

## goal
Upgrade the current static score-artifact flow to a production mainline with runtime auto-fetch, canonical merge, Transformer-style ranking inference, and deterministic Top20/Top3 outputs while preserving existing deploy commands.

## touched files
- PLANS.md
- README.md
- app.py
- src/runtime_history.py
- src/inference.py
- src/fetch_latest.py
- src/normalize_latest.py
- src/history_store.py
- src/build_rank_windows.py
- src/model_transformer.py
- src/train_transformer.py
- scripts/build_deploy_bundle.sh
- configs/predict.yaml
- tests/test_fetch_latest.py
- tests/test_normalize_latest.py
- tests/test_history_store.py
- tests/test_build_rank_windows.py
- tests/test_inference.py
- tests/test_app.py

## invariants
- Keep deploy contract unchanged:
  - `python -m src.runtime_history --input data/processed/history_processed.csv --output data/runtime_history`
  - `bash scripts/build_deploy_bundle.sh deploy_bundle`
- Python 3.9 / Azure Web App compatible.
- Fail fast for missing input/artifact/schema/version mismatch/conflict.
- Inference deterministic (fixed seed, stable sort, explicit tie-breaker).
- No random split; training uses time-series split only.
- Preserve complete 1..80 score chain.
- Top3 diversity is post-ranking rerank.
- Ranking score is not calibrated probability.

## risks
- External source HTML structure can change and break parsers.
- Runtime fetch depends on network/source availability; `/predict` will fail fast on source failures.
- Lightweight numpy transformer is intentionally simple and may underperform a deep-learning implementation.

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
Revert the final commit to restore previous runtime_history + static artifact behavior.

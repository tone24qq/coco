# Plan

## goal
Upgrade the current runtime/deploy baseline to a production-grade, versioned artifact + deterministic inference mainline while preserving the existing deployment contract in `.github/workflows/deployment.yml`.

## touched files
- PLANS.md
- README.md
- app.py
- scripts/build_deploy_bundle.sh
- src/runtime_history.py
- src/inference.py
- tests/test_runtime_history.py
- tests/test_inference.py
- tests/test_app.py

## invariants
- Keep deploy commands unchanged:
  - `python -m src.runtime_history --input data/processed/history_processed.csv --output data/runtime_history`
  - `bash scripts/build_deploy_bundle.sh deploy_bundle`
- Keep Python 3.9 and Azure Web App compatibility.
- Fail fast for missing input/artifact/schema/version mismatch.
- Deterministic ranking: stable sorting with explicit tie-breaker.
- Preserve full 1..80 score chain in runtime artifact and inference response.
- Top3 diversity is post-ranking rerank, not label shortcut.
- Do not present ranking scores as probabilities.

## risks
- Existing runtime artifact from old version will become incompatible and must be rebuilt.
- Strict schema validation may reject malformed upstream files earlier (intended fail-fast behavior).

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
Revert the final commit to restore the previous baseline.

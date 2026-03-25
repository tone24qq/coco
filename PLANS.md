# Plan

## goal
Refactor `/predict` latest fetch policy from first-success short-circuit to multi-source max-latest selection, add `auzo` source/config, preserve existing ranking contract fields, and expose source-level observability/consensus diagnostics without breaking existing endpoints.

## touched files
- PLANS.md
- configs/predict.yaml
- src/fetch_latest.py
- src/inference.py
- tests/test_fetch_latest.py
- tests/test_inference.py
- ARCHITECTURE.md

## invariants
- Keep `GET /predict` route shape and availability unchanged.
- Preserve ranking output semantics (`score_type=ranking_score`, `scores/top20/top3`, `diversity_relaxed`, `drift_metadata`, `stale_issues`, `is_stale`).
- Keep fail-fast behavior for artifact/schema/tensor/version mismatches.
- Keep runtime_history no-retrain behavior.
- Use configured sources only (no hard-coded runtime source list).
- Deterministic tie-breaking for source selection.

## risks
- Source divergence logic could accidentally reject usable data if thresholds are too strict.
- New response fields may require updates in tests that stub inference output.
- Auzo parser may be brittle if source HTML format changes.

## validation steps
- bash scripts/verify_mainline.sh
- flake8 agent.py
- flake8 src tests
- python -m py_compile $(git ls-files '*.py')
- pytest -q
- pytest -q tests/test_fetch_latest.py tests/test_inference.py

## rollback plan
Revert the commit to restore previous first-success fetch strategy and prior `/predict` response shape.

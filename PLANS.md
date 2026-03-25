# Plan

## goal
Make model-ranking outputs explicit and primary (`raw_*` + `final_*`), keep anti-repeat style rules disabled by default, and expose optional rerank impact observability without changing core inference sequence modeling.

## touched files
- PLANS.md
- src/fetch_latest.py
- src/inference.py
- tests/test_fetch_latest.py
- tests/test_inference.py
- README.md
- ARCHITECTURE.md

## invariants
- Keep `GET /predict` route shape and availability unchanged.
- Preserve ranking output semantics (`score_type=ranking_score`, `scores/top20/top3`, `diversity_relaxed`, `drift_metadata`, `stale_issues`, `is_stale`).
- Keep fail-fast behavior for artifact/schema/tensor/version mismatches.
- Keep runtime_history no-retrain behavior.
- Use configured sources only (no hard-coded runtime source list).
- Deterministic tie-breaking for source selection.
- Distinguish `full_records` vs `latest_tail_records` without name ambiguity.
- Fail fast when normalized latest records violate issue-wise row invariants.
- Final ranking must default to raw model ordering with optional rerank clearly flagged.

## risks
- Live source structure drift can reduce parsed full-day issue counts.
- Existing tests may assume `source_records_count` semantics; changing to full-count may require expectation updates.
- PyTorch import limitations in this environment can block full `pytest -q`.
- Additional response fields may require downstream clients to adapt if they validate strict schemas.

## validation steps
- bash scripts/verify_mainline.sh
- flake8 agent.py
- flake8 src tests
- python -m py_compile $(git ls-files '*.py')
- pytest -q
- pytest -q tests/test_fetch_latest.py
- pytest -q tests/test_inference.py
- live fetch integration script for selected source + issue first/last/count output
- live predict-shape script printing prev_draw/raw_top20/final_top20 overlap diagnostics

## rollback plan
Revert this commit to restore previous multi-source behavior (tail-returning semantics).

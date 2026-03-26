# Plan

## goal
Allow `merge_history` to accept non-consecutive issue sequences while preserving strict ordering, duplicate detection, and overlap conflict fail-fast checks.

## touched files
- PLANS.md
- src/history_store.py
- tests/test_history_store.py
- README.md
- ARCHITECTURE.md

## invariants
- Keep canonical history schema checks unchanged.
- Keep issue sequence strictly increasing and duplicate-free.
- Keep fail-fast behavior for overlapping issue conflicts.
- Keep deterministic merge ordering (sorted by issue as integer).
- Keep runtime artifacts and prediction output contract unchanged.

## risks
- Downstream readers may assume issue continuity and interpret gap-based windows as issue distance.
- Existing tests that asserted contiguous issues can fail until updated.
- Full test suite may contain unrelated legacy failures in this environment.

## validation steps
- flake8 src tests
- python -m py_compile $(git ls-files '*.py')
- pytest -q tests/test_history_store.py
- pytest -q

## rollback plan
Revert this commit to restore the previous strict-consecutive merge behavior.

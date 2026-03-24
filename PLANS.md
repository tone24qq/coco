# Plan

## goal
Upgrade Transformer training UX and reliability with Chinese progress output, percentage updates, max-issues time-ordered slicing, and early-stop patience=3, while keeping artifact/output contracts stable; also optimize inference runtime path with safe caching and Chinese per-stage latency logs to meet <=10s SLA target without changing ranking semantics.

## touched files
- PLANS.md
- src/train_transformer.py
- src/build_rank_windows.py
- src/inference.py
- tests/test_train_transformer.py
- tests/test_inference.py

## invariants
- Keep Transformer architecture/output contracts unchanged.
- Keep `model.ckpt` and `transformer_metadata.json` compatibility.
- Keep app -> inference -> output schema unchanged (`scores/top20/top3/score_type`).
- No retraining during inference/runtime history.
- Deterministic inference behavior preserved.

## risks
- Additional logging may increase console noise.
- Caching must respect artifact updates; stale cache risk mitigated via mtime keys.
- PyTorch-heavy tests are slow in this environment.

## validation steps
- python -m pip install -r requirements-dev.txt
- flake8 agent.py
- flake8 src tests
- python -m py_compile $(git ls-files '*.py')
- pytest -q
- python -m src.train_transformer --input <csv> --output <dir> --max-issues 3000
- python -m src.predict --runtime-dir <dir>

## rollback plan
Revert this commit to restore pre-upgrade train/inference behavior.

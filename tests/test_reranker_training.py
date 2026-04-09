from __future__ import annotations

import json
from pathlib import Path


def test_train_reranker_fallback_runs() -> None:
    import scripts.train_reranker as tr

    tr.main()
    weights = Path("artifacts/reranker_weights.json")
    assert weights.exists()
    payload = json.loads(weights.read_text())
    assert "enabled" in payload
    assert "feature_schema_version" in payload

    cv = Path("reports/reranker_cv_summary.json")
    assert cv.exists()

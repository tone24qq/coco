import json
from pathlib import Path

import pandas as pd
import pytest

from src.inference import predict
from src.runtime_history import ARTIFACT_VERSION


def _write_valid_artifact(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "score_artifact": "scores.csv",
        "score_chain_size": 80,
        "history_rows": 2,
        "latest_issue": "1002",
    }
    (runtime_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    scores = pd.DataFrame([{"number": i, "score": float(81 - i)} for i in range(1, 81)])
    scores.to_csv(runtime_dir / "scores.csv", index=False)


def test_predict_loads_artifact_success(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_valid_artifact(runtime_dir)

    result = predict(runtime_dir)
    scores = result["scores"]
    top20 = result["top20"]
    top3 = result["top3"]
    if not isinstance(scores, list) or len(scores) != 80:
        pytest.fail("scores chain must have 80 entries")
    if not isinstance(top20, list) or len(top20) != 20:
        pytest.fail("top20 must have 20 entries")
    if not isinstance(top3, list) or len(top3) != 3:
        pytest.fail("top3 must have 3 entries")


def test_predict_artifact_version_mismatch_fail(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_valid_artifact(runtime_dir)

    metadata_path = runtime_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifact_version"] = "runtime_history_v1"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Artifact version mismatch"):
        predict(runtime_dir)


def test_predict_score_chain_incomplete_fail(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_valid_artifact(runtime_dir)

    scores = pd.DataFrame([{"number": i, "score": float(81 - i)} for i in range(1, 80)])
    scores.to_csv(runtime_dir / "scores.csv", index=False)

    with pytest.raises(ValueError, match="Score chain is not complete"):
        predict(runtime_dir)

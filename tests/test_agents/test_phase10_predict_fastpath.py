import json
from pathlib import Path

import joblib
import yaml
import pytest

from src.artifacts import load_artifacts
from src.modeling import fit_models, load_ranking_dataset, resolve_feature_columns
from src.predict import build_prediction_runtime_state, run_prediction
from src.utils import DataContractError


def _train_artifacts(dataset_path: Path, models_dir: Path) -> None:
    df = load_ranking_dataset(dataset_path)
    cols = resolve_feature_columns(df)
    ranker, logistic = fit_models(df, cols)
    models_dir.mkdir(exist_ok=True)
    ranker.booster_.save_model(str(models_dir / "lightgbm_ranker.txt"))
    joblib.dump(logistic, models_dir / "logistic_regression.pkl")
    (models_dir / "feature_columns.json").write_text(json.dumps(cols), encoding="utf-8")
    (models_dir / "metadata.json").write_text(json.dumps({"model_family": "test", "created_at": "2026"}), encoding="utf-8")


def _config(tmp_path: Path, synthetic_records) -> dict:
    cfg = yaml.safe_load(Path("configs/predict.yaml").read_text(encoding="utf-8"))
    cfg["auto_fetch"]["enabled"] = False
    processed = tmp_path / "history_processed.csv"
    with processed.open("w", encoding="utf-8") as fh:
        fh.write("issue,draw_date,numbers,day_issue_index\n")
        for r in synthetic_records[:-1]:
            fh.write(f"{r.issue},{r.draw_date.isoformat()},\"{json.dumps(list(r.numbers), ensure_ascii=False)}\",{r.day_issue_index}\n")
    cfg["history"]["processed_path"] = str(processed)
    cfg["history"]["runtime_artifact_dir"] = str(tmp_path / "runtime_history")
    cfg["provenance"]["audit_path"] = str(tmp_path / "audit.json")
    cfg["snapshot"]["path"] = str(tmp_path / "snapshot.json")
    return cfg


def test_startup_preload_fail_fast_missing_artifacts(ranking_dataset_path, synthetic_records, tmp_path):
    cfg = _config(tmp_path, synthetic_records)
    models_dir = tmp_path / "missing_models"
    with pytest.raises(DataContractError):
        build_prediction_runtime_state(load_artifacts(models_dir), cfg)


def test_fast_path_reuses_preloaded_history_and_retrieval(monkeypatch, ranking_dataset_path, synthetic_records, tmp_path):
    models_dir = tmp_path / "models"
    _train_artifacts(ranking_dataset_path, models_dir)
    cfg = _config(tmp_path, synthetic_records)
    artifacts = load_artifacts(models_dir)
    state = build_prediction_runtime_state(artifacts, cfg)

    called = {"n": 0}

    def forbidden(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("prepared retrieval should be used on fast path")

    monkeypatch.setattr("src.retrieval.SimilarWindowRetriever.query", forbidden)
    out = run_prediction(artifacts, cfg, response_mode="minimal", runtime_state=state)
    assert len(out["top20_numbers"]) == 20
    assert called["n"] == 0


def test_recent_cache_status_and_repeated_output_metadata(ranking_dataset_path, synthetic_records, tmp_path):
    models_dir = tmp_path / "models"
    _train_artifacts(ranking_dataset_path, models_dir)
    cfg = _config(tmp_path, synthetic_records)
    artifacts = load_artifacts(models_dir)
    state = build_prediction_runtime_state(artifacts, cfg)

    out1 = run_prediction(artifacts, cfg, response_mode="minimal", runtime_state=state)
    out2 = run_prediction(artifacts, cfg, response_mode="minimal", runtime_state=state)
    m1 = out1["metadata"]
    m2 = out2["metadata"]
    assert m1["recent_cache_status"] in {"hit", "miss", "refreshed"}
    assert m2["recent_cache_status"] in {"hit", "miss", "refreshed", "stale"}
    assert m2["top20_jaccard_vs_prev"] is not None
    assert m2["top10_jaccard_vs_prev"] is not None

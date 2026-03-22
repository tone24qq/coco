import json
from pathlib import Path

import pytest
import yaml

from src.predict import _clear_runtime_history_cache, normalize_predict_config_paths, run_prediction
from src.utils import DataContractError


def _train_artifacts(ranking_dataset_path: Path, tmp_path: Path) -> Path:
    import joblib

    from src.modeling import fit_models, load_ranking_dataset, resolve_feature_columns

    df = load_ranking_dataset(ranking_dataset_path)
    cols = resolve_feature_columns(df)
    ranker, logistic = fit_models(df, cols)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    ranker.booster_.save_model(str(models_dir / "lightgbm_ranker.txt"))
    joblib.dump(logistic, models_dir / "logistic_regression.pkl")
    (models_dir / "feature_columns.json").write_text(json.dumps(cols), encoding="utf-8")
    (models_dir / "metadata.json").write_text(json.dumps({"model_family": "test", "created_at": "2026"}), encoding="utf-8")
    return models_dir


def _config(tmp_path: Path, processed_path: Path) -> dict:
    cfg = yaml.safe_load(Path("configs/predict.yaml").read_text(encoding="utf-8"))
    cfg["auto_fetch"]["enabled"] = False
    cfg["history"]["processed_path"] = str(processed_path)
    cfg["history"]["runtime_artifact_dir"] = str(tmp_path / "runtime_history")
    cfg["provenance"] = {
        "audit_path": str(tmp_path / "local_data_audit.json"),
        "consensus_report_path": str(tmp_path / "source_consensus_report.json"),
    }
    cfg["snapshot"] = {"path": str(tmp_path / "history_snapshot.json")}
    return cfg


def _write_processed(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("issue,draw_date,numbers,day_issue_index\n")
        for r in rows:
            fh.write(f"{r.issue},{r.draw_date.isoformat()},\"{json.dumps(list(r.numbers), ensure_ascii=False)}\",{r.day_issue_index}\n")


def test_runtime_artifact_is_deploy_input_and_matches_old_processed_loader(
    monkeypatch, ranking_dataset_path: Path, synthetic_records, tmp_path: Path
) -> None:
    from src.artifacts import load_artifacts
    from src.runtime_history import build_runtime_history_artifact

    models_dir = _train_artifacts(ranking_dataset_path, tmp_path)
    processed = tmp_path / "history_processed.csv"
    _write_processed(processed, synthetic_records[:-2])
    recent = [r.to_dict() for r in synthetic_records[-30:]]
    cfg = _config(tmp_path, processed)

    def old_loader(config):
        from src.utils import read_processed

        return read_processed(Path(config["history"]["processed_path"]))

    monkeypatch.setattr("src.predict._load_runtime_history", old_loader)
    baseline = run_prediction(load_artifacts(models_dir), cfg, recent)

    _clear_runtime_history_cache()
    monkeypatch.undo()
    build_runtime_history_artifact(processed, Path(cfg["history"]["runtime_artifact_dir"]))
    processed.unlink()

    monkeypatch.setattr(
        "src.predict.build_runtime_history_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rebuild must not be called when artifact is ready")),
    )
    current = run_prediction(load_artifacts(models_dir), cfg, recent)

    for key in ["top20_numbers", "top10_numbers", "top3_numbers", "ranking_score_table", "retrieval_top_matches"]:
        assert current[key] == baseline[key]


def test_runtime_does_not_use_pandas_concat_or_raw_rebuild(monkeypatch, ranking_dataset_path: Path, synthetic_records, tmp_path: Path) -> None:
    from src.artifacts import load_artifacts

    models_dir = _train_artifacts(ranking_dataset_path, tmp_path)
    processed_base = tmp_path / "history_processed.csv"
    part1 = tmp_path / "history_processed.part0001.csv"
    part2 = tmp_path / "history_processed.part0002.csv"
    _write_processed(part1, synthetic_records[:100])
    _write_processed(part2, synthetic_records[100:-2])

    cfg = _config(tmp_path, processed_base)
    recent = [r.to_dict() for r in synthetic_records[-30:]]
    _clear_runtime_history_cache()

    monkeypatch.setattr("src.utils.read_csv_maybe_sharded", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("pandas sharded loader forbidden")))
    monkeypatch.setattr("src.io.canonical_dataset.build_canonical_audit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("raw rebuild forbidden")))

    out = run_prediction(load_artifacts(models_dir), cfg, recent)
    assert len(out["top20_numbers"]) == 20


def test_runtime_history_cache_reused(monkeypatch, ranking_dataset_path: Path, synthetic_records, tmp_path: Path) -> None:
    from src.artifacts import load_artifacts

    models_dir = _train_artifacts(ranking_dataset_path, tmp_path)
    processed = tmp_path / "history_processed.csv"
    _write_processed(processed, synthetic_records[:-2])
    cfg = _config(tmp_path, processed)
    recent = [r.to_dict() for r in synthetic_records[-30:]]

    import src.predict as predict_module

    call_count = {"n": 0}
    real_builder = predict_module.build_runtime_history_artifact

    def wrapped_builder(*args, **kwargs):
        call_count["n"] += 1
        return real_builder(*args, **kwargs)

    _clear_runtime_history_cache()
    monkeypatch.setattr(predict_module, "build_runtime_history_artifact", wrapped_builder)

    run_prediction(load_artifacts(models_dir), cfg, recent)
    run_prediction(load_artifacts(models_dir), cfg, recent)
    assert call_count["n"] == 1


def test_fail_fast_when_no_compact_or_processed(ranking_dataset_path: Path, synthetic_records, tmp_path: Path) -> None:
    from src.artifacts import load_artifacts

    models_dir = _train_artifacts(ranking_dataset_path, tmp_path)
    cfg = _config(tmp_path, tmp_path / "missing_history_processed.csv")
    recent = [r.to_dict() for r in synthetic_records[-30:]]

    _clear_runtime_history_cache()
    with pytest.raises(DataContractError, match="runtime history artifact missing and processed history missing; cannot rebuild"):
        run_prediction(load_artifacts(models_dir), cfg, recent)


def test_normalize_predict_config_paths_resolve_absolute(tmp_path: Path) -> None:
    cfg = {
        "models": {"dir": "models"},
        "history": {"processed_path": "data/processed/history_processed.csv", "runtime_artifact_dir": "data/runtime_history"},
        "provenance": {"audit_path": "reports/local_data_audit.json", "raw_dirs": ["data/raw"]},
        "snapshot": {"path": "reports/history_snapshot.json"},
    }
    norm = normalize_predict_config_paths(cfg, base_dir=tmp_path)
    assert Path(norm["models"]["dir"]).is_absolute()
    assert Path(norm["history"]["processed_path"]).is_absolute()
    assert Path(norm["history"]["runtime_artifact_dir"]).is_absolute()
    assert Path(norm["provenance"]["audit_path"]).is_absolute()
    assert all(Path(x).is_absolute() for x in norm["provenance"]["raw_dirs"])
    assert Path(norm["snapshot"]["path"]).is_absolute()

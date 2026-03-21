import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml
import pytest

from src.analysis.snapshots import build_history_snapshot
from src.backtest import main as backtest_main
from src.predict import _clear_runtime_history_cache, run_prediction
from src.utils import DataContractError


def _write_raw_history_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("issue,draw_date,numbers\n")
        for r in rows:
            payload = json.dumps(list(r.numbers), ensure_ascii=False)
            fh.write(f'{r.issue},{r.draw_date.isoformat()},"{payload}"\n')


def test_snapshot_output(synthetic_records, tmp_path: Path) -> None:
    out = tmp_path / "history_snapshot.json"
    snap = build_history_snapshot(synthetic_records, output_path=out)
    assert out.exists()
    assert snap["total_history_rows"] == len(synthetic_records)
    assert "recent_window_summaries" in snap


def test_predict_metadata_extended(ranking_dataset_path, synthetic_records, tmp_path: Path) -> None:
    import joblib

    from src.artifacts import load_artifacts
    from src.modeling import fit_models, load_ranking_dataset, resolve_feature_columns

    df = load_ranking_dataset(ranking_dataset_path)
    cols = resolve_feature_columns(df)
    ranker, logistic = fit_models(df, cols)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ranker.booster_.save_model(str(models_dir / "lightgbm_ranker.txt"))
    joblib.dump(logistic, models_dir / "logistic_regression.pkl")
    (models_dir / "feature_columns.json").write_text(json.dumps(cols), encoding="utf-8")
    (models_dir / "metadata.json").write_text(json.dumps({"model_family": "test", "created_at": "2026"}), encoding="utf-8")

    cfg = yaml.safe_load(Path("configs/predict.yaml").read_text(encoding="utf-8"))
    cfg["auto_fetch"]["enabled"] = False
    processed_path = tmp_path / "history_processed.csv"
    with processed_path.open("w", encoding="utf-8") as fh:
        fh.write("issue,draw_date,numbers,day_issue_index\n")
        for r in synthetic_records[:-2]:
            fh.write(f"{r.issue},{r.draw_date.isoformat()},\"{json.dumps(list(r.numbers), ensure_ascii=False)}\",{r.day_issue_index}\n")
    cfg["history"]["processed_path"] = str(processed_path)
    cfg["provenance"] = {
        "audit_path": str(tmp_path / "local_data_audit.json"),
        "consensus_report_path": str(tmp_path / "source_consensus_report.json"),
    }
    cfg["snapshot"] = {"path": str(tmp_path / "history_snapshot.json")}
    (tmp_path / "local_data_audit.json").write_text(json.dumps({"detected_files": ["x.csv"], "canonical_rows": 100}), encoding="utf-8")

    out = run_prediction(load_artifacts(models_dir), cfg, [r.to_dict() for r in synthetic_records[-30:]])
    assert out["metadata"]["score_type"] == "ranking_score"
    assert out["metadata"]["auxiliary_score"] == "logistic_score"
    assert "history_snapshot" in out["metadata"]
    assert "source_consensus_status" in out["metadata"]
    assert out["metadata"]["runtime_history_rows"] > out["metadata"]["runtime_recent_context_rows"]


def test_predict_fail_fast_on_history_recent_contract_mismatch(ranking_dataset_path, synthetic_records, tmp_path: Path) -> None:
    import joblib

    from src.artifacts import load_artifacts
    from src.modeling import fit_models, load_ranking_dataset, resolve_feature_columns

    df = load_ranking_dataset(ranking_dataset_path)
    cols = resolve_feature_columns(df)
    ranker, logistic = fit_models(df, cols)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ranker.booster_.save_model(str(models_dir / "lightgbm_ranker.txt"))
    joblib.dump(logistic, models_dir / "logistic_regression.pkl")
    (models_dir / "feature_columns.json").write_text(json.dumps(cols), encoding="utf-8")
    (models_dir / "metadata.json").write_text(json.dumps({"model_family": "test", "created_at": "2026"}), encoding="utf-8")

    cfg = yaml.safe_load(Path("configs/predict.yaml").read_text(encoding="utf-8"))
    cfg["auto_fetch"]["enabled"] = False
    processed_path = tmp_path / "history_processed.csv"
    with processed_path.open("w", encoding="utf-8") as fh:
        fh.write("issue,draw_date,numbers,day_issue_index\n")
        for r in synthetic_records[:-1]:
            fh.write(f"{r.issue},{r.draw_date.isoformat()},\"{json.dumps(list(r.numbers), ensure_ascii=False)}\",{r.day_issue_index}\n")
    cfg["history"]["processed_path"] = str(processed_path)
    cfg["provenance"] = {
        "audit_path": str(tmp_path / "local_data_audit.json"),
        "consensus_report_path": str(tmp_path / "source_consensus_report.json"),
    }
    (tmp_path / "local_data_audit.json").write_text(json.dumps({"detected_files": ["x.csv"], "canonical_rows": 100}), encoding="utf-8")

    bad_recent = [r.to_dict() for r in synthetic_records[-3:]]
    bad_recent[0]["numbers"] = list(range(1, 21))
    with pytest.raises(DataContractError):
        run_prediction(load_artifacts(models_dir), cfg, bad_recent)


def test_backtest_extra_reports_exist(ranking_dataset_path, tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"n_splits": 2, "min_train_issues": 10}, "runtime_scoring": {"weights": {}}}),
        encoding="utf-8",
    )

    fake_val = pd.DataFrame(
        {
            "issue": ["I1", "I1", "I2", "I2"],
            "candidate_number": [1, 2, 1, 2],
            "label": [1, 0, 0, 1],
            "final_score": [0.9, 0.1, 0.2, 0.8],
            "ranker_score": [0.9, 0.1, 0.2, 0.8],
            "logistic_score": [0.8, 0.2, 0.3, 0.7],
            "retrieval_score": [0.7, 0.3, 0.3, 0.7],
            "history_prior_score": [0.6, 0.4, 0.4, 0.6],
            "analysis_rerank_score": [0.5, 0.5, 0.5, 0.5],
            "local_peak_score": [0.5, 0.5, 0.5, 0.5],
            "cand_hits_last_100": [10, 1, 1, 10],
            "cand_hits_last_20": [3, 1, 1, 3],
            "retrieval_top3_hit_flag": [1, 0, 0, 1],
            "retrieval_exact_window_match_count": [0, 0, 0, 0],
            "retrieval_exact_draw_match_count_mean": [1, 0, 0, 1],
        }
    )

    fake_train = fake_val.copy()

    def fake_run_cv(*args, **kwargs):
        return [SimpleNamespace(fold_id=1, val_scored=fake_val.copy(), train_scored=fake_train.copy(), train_issues=["I0"], val_issues=["I1", "I2"])]

    monkeypatch.setattr("src.backtest.run_cv", fake_run_cv)
    monkeypatch.chdir(tmp_path)
    exp_path = tmp_path / "experiments.yaml"
    exp_path.write_text(yaml.safe_dump({"experiments": [{"name": "baseline_frequency"}, {"name": "dynamic_n_fusion_main"}]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["backtest", "--config", str(cfg_path), "--experiments", str(exp_path), "--input", str(ranking_dataset_path)])
    backtest_main()
    assert Path("reports/predictability_test.json").exists()
    assert Path("reports/permutation_distribution.csv").exists()
    assert Path("reports/block_bootstrap_summary.json").exists()
    assert Path("reports/alignment_audit.json").exists()
    summary = json.loads(Path("reports/backtest_experiment_summary.json").read_text(encoding="utf-8"))
    assert summary["train_vs_backtest_gap_top3"] != "unavailable"
    bootstrap = json.loads(Path("reports/block_bootstrap_summary.json").read_text(encoding="utf-8"))
    assert bootstrap["metric"] == "mainline_minus_baseline_top3"


def test_predict_with_sharded_processed_history(ranking_dataset_path, synthetic_records, tmp_path: Path) -> None:
    import joblib

    from src.artifacts import load_artifacts
    from src.modeling import fit_models, load_ranking_dataset, resolve_feature_columns

    df = load_ranking_dataset(ranking_dataset_path)
    cols = resolve_feature_columns(df)
    ranker, logistic = fit_models(df, cols)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ranker.booster_.save_model(str(models_dir / "lightgbm_ranker.txt"))
    joblib.dump(logistic, models_dir / "logistic_regression.pkl")
    (models_dir / "feature_columns.json").write_text(json.dumps(cols), encoding="utf-8")
    (models_dir / "metadata.json").write_text(json.dumps({"model_family": "test", "created_at": "2026"}), encoding="utf-8")

    processed_base = tmp_path / "history_processed.csv"
    part1 = tmp_path / "history_processed.part0001.csv"
    part2 = tmp_path / "history_processed.part0002.csv"
    header = "issue,draw_date,numbers,day_issue_index\n"
    rows = []
    for r in synthetic_records[:-2]:
        rows.append(f"{r.issue},{r.draw_date.isoformat()},\"{json.dumps(list(r.numbers), ensure_ascii=False)}\",{r.day_issue_index}\n")
    split = len(rows) // 2
    part1.write_text(header + "".join(rows[:split]), encoding="utf-8")
    part2.write_text(header + "".join(rows[split:]), encoding="utf-8")

    cfg = yaml.safe_load(Path("configs/predict.yaml").read_text(encoding="utf-8"))
    cfg["auto_fetch"]["enabled"] = False
    cfg["history"]["processed_path"] = str(processed_base)
    cfg["provenance"] = {
        "audit_path": str(tmp_path / "local_data_audit.json"),
        "consensus_report_path": str(tmp_path / "source_consensus_report.json"),
    }
    cfg["snapshot"] = {"path": str(tmp_path / "history_snapshot.json")}

    out = run_prediction(load_artifacts(models_dir), cfg, [r.to_dict() for r in synthetic_records[-30:]])
    assert out["metadata"]["runtime_history_rows"] >= len(synthetic_records[:-2])


def test_predict_fail_fast_when_processed_and_raw_missing(ranking_dataset_path, synthetic_records, tmp_path: Path) -> None:
    import joblib

    from src.artifacts import load_artifacts
    from src.modeling import fit_models, load_ranking_dataset, resolve_feature_columns

    df = load_ranking_dataset(ranking_dataset_path)
    cols = resolve_feature_columns(df)
    ranker, logistic = fit_models(df, cols)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ranker.booster_.save_model(str(models_dir / "lightgbm_ranker.txt"))
    joblib.dump(logistic, models_dir / "logistic_regression.pkl")
    (models_dir / "feature_columns.json").write_text(json.dumps(cols), encoding="utf-8")
    (models_dir / "metadata.json").write_text(json.dumps({"model_family": "test", "created_at": "2026"}), encoding="utf-8")

    cfg = yaml.safe_load(Path("configs/predict.yaml").read_text(encoding="utf-8"))
    cfg["auto_fetch"]["enabled"] = False
    cfg["history"]["processed_path"] = str(tmp_path / "missing_processed.csv")
    cfg["history"]["runtime_artifact_dir"] = str(tmp_path / "missing_runtime_history")
    cfg["provenance"] = {
        "raw_dirs": [str(tmp_path / "raw")],
        "audit_path": str(tmp_path / "local_data_audit.json"),
        "manifest_path": str(tmp_path / "raw_manifest.json"),
        "consensus_report_path": str(tmp_path / "source_consensus_report.json"),
    }

    _clear_runtime_history_cache()
    with pytest.raises(DataContractError, match="runtime history artifact missing and processed history missing; cannot rebuild"):
        run_prediction(load_artifacts(models_dir), cfg, [r.to_dict() for r in synthetic_records[-30:]])

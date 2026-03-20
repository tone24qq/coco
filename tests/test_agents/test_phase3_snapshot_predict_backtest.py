import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from src.analysis.snapshots import build_history_snapshot
from src.backtest import main as backtest_main
from src.predict import run_prediction


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

    def fake_run_cv(*args, **kwargs):
        return [SimpleNamespace(fold_id=1, val_scored=fake_val.copy())]

    monkeypatch.setattr("src.backtest.run_cv", fake_run_cv)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["backtest", "--config", str(cfg_path), "--input", str(ranking_dataset_path)])
    backtest_main()
    assert Path("reports/predictability_test.json").exists()
    assert Path("reports/permutation_distribution.csv").exists()
    assert Path("reports/block_bootstrap_summary.json").exists()
    assert Path("reports/alignment_audit.json").exists()

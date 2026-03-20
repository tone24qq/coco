import json
from pathlib import Path

import joblib
import yaml

from src.artifacts import load_artifacts
from src.modeling import fit_models, load_ranking_dataset, resolve_feature_columns
from src.predict import run_prediction


def _train_artifacts(dataset_path: Path, models_dir: Path) -> None:
    df = load_ranking_dataset(dataset_path)
    cols = resolve_feature_columns(df)
    ranker, logistic = fit_models(df, cols)
    models_dir.mkdir(exist_ok=True)
    ranker.booster_.save_model(str(models_dir / "lightgbm_ranker.txt"))
    joblib.dump(logistic, models_dir / "logistic_regression.pkl")
    (models_dir / "feature_columns.json").write_text(json.dumps(cols), encoding="utf-8")
    (models_dir / "metadata.json").write_text(
        json.dumps({"model_family": "test", "created_at": "2026-01-01T00:00:00"}), encoding="utf-8"
    )


def test_predict_output_schema(ranking_dataset_path, synthetic_records, tmp_path) -> None:
    models_dir = tmp_path / "models"
    _train_artifacts(ranking_dataset_path, models_dir)
    cfg = yaml.safe_load(Path("configs/predict.yaml").read_text(encoding="utf-8"))
    processed = tmp_path / "history_processed.csv"
    with processed.open("w", encoding="utf-8") as fh:
        fh.write("issue,draw_date,numbers,day_issue_index\n")
        for row in synthetic_records[:-1]:
            fh.write(f"{row.issue},{row.draw_date.isoformat()},\"{json.dumps(list(row.numbers), ensure_ascii=False)}\",{row.day_issue_index}\n")
    cfg["history"]["processed_path"] = str(processed)
    artifacts = load_artifacts(models_dir)
    recent = [r.to_dict() for r in synthetic_records[-30:]]
    out = run_prediction(artifacts, cfg, recent)
    assert len(out["top20_numbers"]) == 20
    assert len(out["ranking_score_table"]) == 80
    assert "ranker_score" in out["ranking_score_table"][0]
    assert "candidate_number" in out["ranking_score_table"][0]
    assert sorted(int(r["candidate_number"]) for r in out["ranking_score_table"]) == list(range(1, 81))
    assert out["issue"] == str(int(synthetic_records[-1].issue) + 1)
    assert out["dynamic_context_n"] >= 20

import json
from pathlib import Path

import joblib
import yaml

from src.build_features import build_feature_rows
from src.modeling import fit_models, load_ranking_dataset, resolve_feature_columns
from src.predict import run_prediction


def test_history_to_next_issue_label_contract(synthetic_records) -> None:
    rows = build_feature_rows(synthetic_records, min_history=100, min_dynamic_n=20, max_dynamic_n=40, top_k=8)
    first_issue = rows[0]["issue"]
    index_by_issue = {r.issue: i for i, r in enumerate(synthetic_records)}
    idx = index_by_issue[first_issue]
    next_numbers = set(synthetic_records[idx + 1].numbers)
    first_issue_rows = [r for r in rows if r["issue"] == first_issue]
    assert len(first_issue_rows) == 80
    positives = {int(r["candidate_number"]) for r in first_issue_rows if int(r["label"]) == 1}
    assert positives == next_numbers


def test_predict_ranking_score_table_and_runtime_score_reality(ranking_dataset_path, synthetic_records, tmp_path: Path) -> None:
    df = load_ranking_dataset(ranking_dataset_path)
    cols = resolve_feature_columns(df)
    ranker, logistic = fit_models(df, cols)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ranker.booster_.save_model(str(models_dir / "lightgbm_ranker.txt"))
    joblib.dump(logistic, models_dir / "logistic_regression.pkl")
    (models_dir / "feature_columns.json").write_text(json.dumps(cols), encoding="utf-8")
    (models_dir / "metadata.json").write_text(json.dumps({"model_family": "test", "created_at": "2026"}), encoding="utf-8")

    processed = tmp_path / "history_processed.csv"
    with processed.open("w", encoding="utf-8") as fh:
        fh.write("issue,draw_date,numbers,day_issue_index\n")
        for row in synthetic_records[:-1]:
            fh.write(f"{row.issue},{row.draw_date.isoformat()},\"{json.dumps(list(row.numbers), ensure_ascii=False)}\",{row.day_issue_index}\n")

    cfg = yaml.safe_load(Path("configs/predict.yaml").read_text(encoding="utf-8"))
    cfg["models"] = {"dir": str(models_dir)}
    cfg["history"]["processed_path"] = str(processed)

    out = run_prediction(
        artifacts=type("Artifacts", (), {
            "ranker": ranker,
            "logistic": logistic,
            "feature_columns": cols,
            "metadata": {"model_family": "test", "created_at": "2026"},
        })(),
        config=cfg,
        recent_draws=[r.to_dict() for r in synthetic_records[-30:]],
    )

    table = out["ranking_score_table"]
    assert len(table) == 80
    assert sorted(int(r["candidate_number"]) for r in table) == list(range(1, 81))
    # final_score must be from multi-score chain, not a plain copy of one score column
    assert any(abs(r["final_score"] - r["ranker_score"]) > 1e-9 for r in table)
    assert any(abs(r["final_score"] - r["logistic_score"]) > 1e-9 for r in table)

    sorted_by_rank = sorted(table, key=lambda r: r["rank_final"])
    assert out["top20_numbers"] == [int(r["candidate_number"]) for r in sorted_by_rank[:20]]
    assert out["top10_numbers"] == out["top20_numbers"][:10]


def test_verify_mainline_has_file_size_gate() -> None:
    content = Path("scripts/verify_mainline.sh").read_text(encoding="utf-8")
    assert "st_size" in content
    assert "100 * 1024 * 1024" in content

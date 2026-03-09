from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
from dataclasses import dataclass

import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import (  # noqa: E402
    CONFIG_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    build_candidate_matrix,
    build_latest_issue_features_for_inference,
    classify_board,
    compact_10_from_top20,
    load_yaml,
    zone_of,
)


@dataclass
class Predictor:
    model: lgb.Booster
    feature_columns: list[str]

    @classmethod
    def load(cls) -> "Predictor":
        model = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_top20.txt"))
        cols = json.loads(
            (MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8")
        )
        return cls(model=model, feature_columns=cols)

    def predict_from_draws(self, draws_df: pd.DataFrame, min_history: int) -> dict:
        issue_df = build_latest_issue_features_for_inference(
            draws_df, min_history=min_history
        )
        if issue_df.empty:
            raise ValueError("not enough history for feature generation")
        row = issue_df.iloc[-1]
        x = build_candidate_matrix(row, self.feature_columns)
        scores = self.model.predict(x)
        score_table = pd.DataFrame(
            {"number": list(range(1, 81)), "score": scores}
        ).sort_values("score", ascending=False)
        top20 = score_table["number"].head(20).astype(int).tolist()
        compact10 = compact_10_from_top20(top20)
        top3 = top20[:3]
        latest_issue = int(row["issue"])
        zc = {z: sum(1 for n in top20 if zone_of(n) == z) for z in ["A", "B", "C", "D"]}
        board_type = classify_board(zc)
        raw_score_table = score_table.to_dict(orient="records")
        calibrated_probability_table = [
            {"number": x["number"], "probability": x["score"]} for x in raw_score_table
        ]
        return {
            "target_issue": latest_issue + 1,
            "top20_numbers": top20,
            "compact10_numbers": compact10,
            "top3_core_group": top3,
            "raw_score_table": raw_score_table,
            "calibrated_probability_table": calibrated_probability_table,
            "score_table": raw_score_table,
            "board_type_prediction": board_type,
        }


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "predict.yaml")
    predictor = Predictor.load()
    df = (
        pd.read_csv(DATA_PROCESSED_DIR / "bingo_draws.csv")
        .tail(int(cfg.get("recent_draws_limit", 3000)))
        .reset_index(drop=True)
    )
    print(
        json.dumps(
            predictor.predict_from_draws(
                df, min_history=int(cfg["feature_min_history"])
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

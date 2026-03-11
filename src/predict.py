from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.strategy import StrategyConfig, apply_strategy, derive_regime  # noqa: E402
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


def _load_strategy_payload() -> dict:
    strategy_path = MODELS_DIR / "strategy_config.json"
    if strategy_path.exists():
        return json.loads(strategy_path.read_text(encoding="utf-8"))
    metadata_path = MODELS_DIR / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


@dataclass
class Predictor:
    model: CatBoostClassifier
    feature_columns: list[str]
    strategy: StrategyConfig

    @classmethod
    def load(cls) -> "Predictor":
        model = CatBoostClassifier()
        model.load_model(str(MODELS_DIR / "catboost_top20.cbm"))
        cols = json.loads(
            (MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8")
        )
        strategy_payload = _load_strategy_payload()
        strat = (
            strategy_payload.get("selected_strategy")
            or strategy_payload.get("fallback_strategy")
            or {}
        )
        strategy = StrategyConfig(
            version_id=strat.get("version_id", "v0_binary_baseline"),
            stage_type=strat.get("stage_type", "baseline"),
            candidate_pool=int(strat.get("candidate_pool", 20)),
            prior_window=int(strat.get("prior_window", 100)),
            rerank_weight=float(strat.get("rerank_weight", 0.0)),
            penalty_weight=float(strat.get("penalty_weight", 0.0)),
            trend_weight=float(strat.get("trend_weight", 0.0)),
            regime_aware=bool(strat.get("regime_aware", False)),
        )
        return cls(model=model, feature_columns=cols, strategy=strategy)

    def predict_from_draws(self, draws_df: pd.DataFrame, min_history: int) -> dict:
        issue_df = build_latest_issue_features_for_inference(
            draws_df, min_history=min_history
        )
        if issue_df.empty:
            raise ValueError("not enough history for feature generation")
        row = issue_df.iloc[-1]
        x = build_candidate_matrix(row, self.feature_columns).reindex(
            columns=self.feature_columns
        )
        base_scores = self.model.predict_proba(x)[:, 1]
        regime = derive_regime(row)
        scores = apply_strategy(base_scores, x, self.strategy, regime)

        score_table = pd.DataFrame(
            {"number": list(range(1, 81)), "score": scores}
        ).sort_values("score", ascending=False)
        top20 = score_table["number"].head(20).astype(int).tolist()
        compact10 = compact_10_from_top20(top20)
        top10 = top20[:10]
        top3 = top20[:3]
        latest_issue = int(row["issue"])
        zc = {z: sum(1 for n in top20 if zone_of(n) == z) for z in ["A", "B", "C", "D"]}
        board_type = classify_board(zc)
        raw_score_table = score_table.to_dict(orient="records")
        top20_scores = {
            f"{int(rec['number']):02d}": float(rec["score"])
            for rec in raw_score_table[:20]
        }
        big_count = sum(1 for n in top20 if n >= 41)
        odd_count = sum(1 for n in top20 if n % 2 == 1)
        return {
            "model": "catboost",
            "strategy_version": self.strategy.version_id,
            "target_issue": latest_issue + 1,
            "top20_numbers": top20,
            "top10_numbers": top10,
            "top3_numbers": top3,
            "top20_scores": top20_scores,
            "compact10_numbers": compact10,
            "top3_core_group": top3,
            "raw_score_table": raw_score_table,
            "calibrated_probability_table": [
                {"number": x["number"], "probability": x["score"]}
                for x in raw_score_table
            ],
            "score_table": raw_score_table,
            "board_type_prediction": board_type,
            "big_count": big_count,
            "small_count": 20 - big_count,
            "size_summary": f"大{big_count} / 小{20 - big_count}",
            "odd_count": odd_count,
            "even_count": 20 - odd_count,
            "odd_even_summary": f"單{odd_count} / 雙{20 - odd_count}",
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

from __future__ import annotations

import json
import logging
import os
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
    classify_feature_mode,
    compact_10_from_top20,
    load_yaml,
    normalize_feature_version,
    resolve_effective_windows,
    validate_feature_columns_contract,
    zone_of,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class Predictor:
    model: CatBoostClassifier
    feature_columns: list[str]
    strategy: StrategyConfig
    feature_version: str
    runtime_config: dict

    @classmethod
    def load(cls) -> "Predictor":
        model = CatBoostClassifier()
        model.load_model(str(MODELS_DIR / "catboost_top20.cbm"))
        cols = json.loads(
            (MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (MODELS_DIR / "metadata.json").read_text(encoding="utf-8")
        )
        metadata_feature_version = normalize_feature_version(
            metadata.get("feature_version", "v3_core20")
        )
        if metadata_feature_version != "v3_core20":
            raise ValueError(
                "unsupported model metadata feature_version; only v3_core20 is supported"
            )
        validate_feature_columns_contract(cols, metadata_feature_version)
        yaml_cfg = load_yaml(CONFIG_DIR / "train.yaml")
        yaml_feature_version = normalize_feature_version(
            yaml_cfg.get("feature_version", "v3_core20")
        )
        if yaml_feature_version != metadata_feature_version:
            LOGGER.warning(
                "predict runtime feature_version mismatch: metadata=%s yaml=%s, using metadata",
                metadata_feature_version,
                yaml_feature_version,
            )
        runtime_cfg = dict(metadata.get("runtime_config", {}))
        runtime_cfg.setdefault("feature_version", metadata_feature_version)
        strategy_cfg_path = MODELS_DIR / "strategy_config.json"
        strategy_cfg = (
            json.loads(strategy_cfg_path.read_text(encoding="utf-8"))
            if strategy_cfg_path.exists()
            else {}
        )
        strat = (
            strategy_cfg.get("selected_strategy")
            or metadata.get("selected_strategy")
            or strategy_cfg.get("fallback_strategy")
            or metadata.get("fallback_strategy")
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
        return cls(
            model=model,
            feature_columns=cols,
            strategy=strategy,
            feature_version=metadata_feature_version,
            runtime_config=runtime_cfg,
        )

    def predict_from_draws(self, draws_df: pd.DataFrame, min_history: int) -> dict:
        history_len = len(draws_df)
        effective_windows = resolve_effective_windows(history_len, self.runtime_config)
        configured_windows = self.runtime_config.get("core_windows", {})
        degraded_features = [
            key
            for key, configured in configured_windows.items()
            if int(configured) > int(effective_windows.get(key, configured))
        ]
        prev_runtime = os.getenv("FEATURE_RUNTIME_CONFIG_JSON")
        prev_version = os.getenv("FEATURE_VERSION_OVERRIDE")
        try:
            os.environ["FEATURE_RUNTIME_CONFIG_JSON"] = json.dumps(
                self.runtime_config,
                ensure_ascii=False,
            )
            os.environ["FEATURE_VERSION_OVERRIDE"] = self.feature_version
            issue_df = build_latest_issue_features_for_inference(
                draws_df, min_history=min_history
            )
            if issue_df.empty:
                raise ValueError("not enough history for feature generation")
            row = issue_df.iloc[-1]
            x = build_candidate_matrix(
                row,
                self.feature_columns,
                strict_features=False,
            ).reindex(columns=self.feature_columns)
        finally:
            if prev_runtime is None:
                os.environ.pop("FEATURE_RUNTIME_CONFIG_JSON", None)
            else:
                os.environ["FEATURE_RUNTIME_CONFIG_JSON"] = prev_runtime
            if prev_version is None:
                os.environ.pop("FEATURE_VERSION_OVERRIDE", None)
            else:
                os.environ["FEATURE_VERSION_OVERRIDE"] = prev_version
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
            "history_length_used": history_len,
            "feature_mode": classify_feature_mode(history_len),
            "degraded_features": degraded_features,
            "effective_windows": effective_windows,
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

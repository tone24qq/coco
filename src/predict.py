from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.features import (  # noqa: E402
    candidate_analysis_compatibility_score,
    derive_analysis_target_profile,
)
from src.analysis.snapshots import load_history_snapshot_payload  # noqa: E402
from src.artifacts import load_cascade_artifacts  # noqa: E402
from src.pipeline import CascadePipeline  # noqa: E402
from src.runtime_scoring import score_candidates_runtime  # noqa: E402
from src.strategy import (  # noqa: E402
    StrategyConfig,
    apply_strategy,
    derive_regime,
    is_cascade_strategy,
)
from src.utils import (  # noqa: E402
    CONFIG_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    V3_CORE20_COLUMNS,
    build_candidate_matrix,
    build_latest_issue_features_for_inference,
    classify_board,
    classify_feature_mode,
    compact_10_from_top20,
    load_yaml,
    normalize_feature_version,
    normalize_pipeline_version,
    resolve_effective_windows,
    validate_feature_columns_contract,
    zone_of,
)

LOGGER = logging.getLogger(__name__)


def _normalize_series(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    denom = float(arr.std(ddof=0))
    if denom <= 1e-9:
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return (arr - float(arr.mean())) / denom


def _normalize_rank_pct(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    if len(arr) == 0:
        return arr
    ranked = arr.rank(method="average", pct=True)
    return ranked.astype(float)


def _history_prior_from_snapshot(
    score_table: pd.DataFrame, snapshot_payload: dict
) -> pd.DataFrame:
    out = score_table.copy()
    priors = snapshot_payload.get("number_priors")
    if priors is None or priors.empty:
        out["history_prior_score"] = 0.0
        return out

    cols = [
        "total_hits_all_time",
        "hits_last_200",
        "hits_last_500",
        "hits_last_1000",
        "today_hits",
        "carryover_from_prev",
        "pm1_neighbor_hits",
        "pm2_neighbor_hits",
        "current_gap_all",
        "avg_gap_all",
        "max_gap_all",
    ]
    priors_df = priors.reset_index(drop=True)
    merged = out.merge(priors_df[["number", *cols]], on="number", how="left")
    merged = merged.fillna(0.0)

    positive = (
        0.28 * _normalize_series(merged["total_hits_all_time"])
        + 0.18 * _normalize_series(merged["hits_last_200"])
        + 0.12 * _normalize_series(merged["hits_last_500"])
        + 0.10 * _normalize_series(merged["hits_last_1000"])
        + 0.07 * _normalize_series(merged["today_hits"])
        + 0.08 * _normalize_series(merged["carryover_from_prev"])
        + 0.09 * _normalize_series(merged["pm1_neighbor_hits"])
        + 0.08 * _normalize_series(merged["pm2_neighbor_hits"])
    )
    penalty = (
        0.30 * _normalize_series(merged["current_gap_all"])
        + 0.10 * _normalize_series(merged["avg_gap_all"])
        + 0.05 * _normalize_series(merged["max_gap_all"])
    )
    merged["history_prior_score"] = (positive - penalty).astype(float)
    return merged


def _analysis_rerank(
    score_table: pd.DataFrame,
    recent_draws: list[list[int]],
    board_priors: dict,
    top_k: int,
    rerank_weight: float,
) -> tuple[pd.DataFrame, dict]:
    work = score_table.copy()
    work["analysis_compatibility_score"] = 0.0
    work["analysis_rerank_score"] = 0.0
    work["score_before_analysis_rerank"] = work["final_score"].astype(float)

    profile = derive_analysis_target_profile(recent_draws, board_priors=board_priors)
    top_k = max(3, min(int(top_k), len(work)))

    top = work.head(top_k).copy()
    top["analysis_compatibility_score"] = top["number"].apply(
        lambda n: candidate_analysis_compatibility_score(int(n), profile)
    )
    top["analysis_rerank_score"] = float(rerank_weight) * top[
        "analysis_compatibility_score"
    ].astype(float)
    top["final_score"] = (
        top["score_before_analysis_rerank"] + top["analysis_rerank_score"]
    )
    top = top.sort_values("final_score", ascending=False)

    tail = work.iloc[top_k:].copy()
    reranked = pd.concat([top, tail], ignore_index=True)
    reranked["score_after_analysis_rerank"] = reranked["final_score"].astype(float)
    reranked = reranked.sort_values("final_score", ascending=False).reset_index(
        drop=True
    )

    summary = {
        "enabled": True,
        "top_k": int(top_k),
        "weight": float(rerank_weight),
        "target_profile": profile,
        "top_k_preview": reranked.head(5)[
            [
                "number",
                "score_before_analysis_rerank",
                "analysis_compatibility_score",
                "analysis_rerank_score",
                "score_after_analysis_rerank",
            ]
        ].to_dict(orient="records"),
    }
    return reranked, summary


def _strategy_from_dict(raw: dict) -> StrategyConfig:
    return StrategyConfig(
        version_id=raw.get("version_id", "v0_binary_baseline"),
        stage_type=raw.get("stage_type", "baseline"),
        candidate_pool=int(raw.get("candidate_pool", 20)),
        prior_window=int(raw.get("prior_window", 100)),
        rerank_weight=float(raw.get("rerank_weight", 0.0)),
        penalty_weight=float(raw.get("penalty_weight", 0.0)),
        trend_weight=float(raw.get("trend_weight", 0.0)),
        regime_aware=bool(raw.get("regime_aware", False)),
        pipeline_version=normalize_pipeline_version(
            raw.get("pipeline_version", "baseline_flat_score")
        ),
        model_artifact_dir=str(raw.get("model_artifact_dir", "")),
        stage1_keep=int(raw.get("stage1_keep", 30)),
        stage2_keep=int(raw.get("stage2_keep", 10)),
    )


def resolve_runtime_strategy(
    predict_cfg: dict,
    strategy_cfg: dict,
    metadata: dict,
    train_cfg: dict | None = None,
) -> tuple[StrategyConfig, str]:
    strat_raw = (
        strategy_cfg.get("selected_strategy")
        or metadata.get("selected_strategy")
        or strategy_cfg.get("fallback_strategy")
        or metadata.get("fallback_strategy")
        or {}
    )
    base_strategy = _strategy_from_dict(strat_raw)

    pipeline_cfg = predict_cfg.get("pipeline", {})
    cfg_pipeline_version = str(pipeline_cfg.get("version", "auto"))
    if cfg_pipeline_version in {"", "auto"}:
        if strat_raw:
            return base_strategy, "strategy_config/metadata"
        train_pipeline = normalize_pipeline_version(
            (train_cfg or {}).get("pipeline", {}).get("version", "baseline_flat_score")
        )
        if train_pipeline.startswith("cascade"):
            fallback = StrategyConfig(
                **{
                    **base_strategy.__dict__,
                    "version_id": "cascade_v1_flow",
                    "stage_type": "cascade",
                    "pipeline_version": train_pipeline,
                    "model_artifact_dir": str(
                        predict_cfg.get("pipeline", {}).get(
                            "artifact_dir", f"models/{train_pipeline}"
                        )
                    ),
                    "stage1_keep": int(
                        predict_cfg.get("pipeline", {}).get("stage1_keep", 30)
                    ),
                    "stage2_keep": int(
                        predict_cfg.get("pipeline", {}).get("stage2_keep", 10)
                    ),
                }
            )
            return fallback, "train.yaml pipeline fallback"
        return base_strategy, "defaults"

    pver = normalize_pipeline_version(cfg_pipeline_version)
    if pver.startswith("cascade"):
        override = StrategyConfig(
            **{
                **base_strategy.__dict__,
                "version_id": str(pipeline_cfg.get("version_id", "cascade_v1_flow")),
                "stage_type": "cascade",
                "pipeline_version": pver,
                "model_artifact_dir": str(
                    pipeline_cfg.get("artifact_dir", f"models/{pver}")
                ),
                "stage1_keep": int(pipeline_cfg.get("stage1_keep", 30)),
                "stage2_keep": int(pipeline_cfg.get("stage2_keep", 10)),
            }
        )
        return override, "predict.yaml pipeline override"

    # baseline_flat_score forces legacy strategy
    legacy_raw = strategy_cfg.get("fallback_strategy") or metadata.get(
        "fallback_strategy", {}
    )
    legacy_strategy = _strategy_from_dict(
        legacy_raw
        or {
            "version_id": "v0_binary_baseline",
            "stage_type": "baseline",
            "pipeline_version": "baseline_flat_score",
        }
    )
    return legacy_strategy, "predict.yaml pipeline override"


@dataclass
class Predictor:
    model: CatBoostClassifier | None
    soft_model: CatBoostRegressor | None
    pm1_model: CatBoostClassifier | None
    feature_columns: list[str]
    strategy: StrategyConfig
    feature_version: str
    runtime_config: dict
    cascade_pipeline: CascadePipeline | None = None

    @classmethod
    def load(cls) -> "Predictor":
        predict_cfg = load_yaml(CONFIG_DIR / "predict.yaml")
        model_path = MODELS_DIR / "catboost_top20.cbm"
        model = CatBoostClassifier()
        soft_model = CatBoostRegressor()
        pm1_model = CatBoostClassifier()
        cols: list[str] = []
        feature_cols_path = MODELS_DIR / "feature_columns.json"
        if feature_cols_path.exists():
            cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
        if model_path.exists():
            model.load_model(str(model_path))
        soft_model_path = MODELS_DIR / "catboost_soft_label.cbm"
        pm1_model_path = MODELS_DIR / "catboost_pm1_proximity.cbm"
        if soft_model_path.exists():
            try:
                soft_model.load_model(str(soft_model_path))
            except Exception as exc:
                LOGGER.warning("failed to load soft model: %s", exc)
                soft_model = None
        else:
            soft_model = None
        if pm1_model_path.exists():
            try:
                pm1_model.load_model(str(pm1_model_path))
            except Exception as exc:
                LOGGER.warning("failed to load pm1 proximity model: %s", exc)
                pm1_model = None
        else:
            pm1_model = None
        metadata_path = MODELS_DIR / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {"feature_version": "v3_core20"}
        )
        metadata_feature_version = normalize_feature_version(
            metadata.get("feature_version", "v3_core20")
        )
        if metadata_feature_version != "v3_core20":
            raise ValueError(
                "unsupported model metadata feature_version; only v3_core20 is supported"
            )
        if cols:
            validate_feature_columns_contract(
                cols, metadata_feature_version, allow_legacy_subset=True
            )
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
        for key in [
            "history_prior",
            "analysis_rerank",
            "long_feature_injection",
            "neighbor_peak_correction",
            "topk_group_dedup",
            "soft_label_training",
            "proximity_model",
        ]:
            if key in predict_cfg:
                runtime_cfg[key] = predict_cfg.get(key, {})
        runtime_cfg.setdefault("feature_version", metadata_feature_version)
        strategy_cfg_path = MODELS_DIR / "strategy_config.json"
        strategy_cfg = (
            json.loads(strategy_cfg_path.read_text(encoding="utf-8"))
            if strategy_cfg_path.exists()
            else {}
        )
        strategy, source = resolve_runtime_strategy(
            predict_cfg,
            strategy_cfg,
            metadata,
            train_cfg=yaml_cfg,
        )
        LOGGER.info(
            "predict strategy resolved from %s: version=%s stage=%s pipeline=%s",
            source,
            strategy.version_id,
            strategy.stage_type,
            strategy.pipeline_version,
        )
        if (
            (not model_path.exists())
            and (not is_cascade_strategy(strategy))
            and (not cols)
        ):
            raise ValueError("legacy model artifact missing: models/catboost_top20.cbm")
        cascade_pipeline = None
        if is_cascade_strategy(strategy):
            artifact_dir = (
                PROJECT_ROOT / strategy.model_artifact_dir
                if strategy.model_artifact_dir
                else MODELS_DIR / strategy.pipeline_version
            )
            if not artifact_dir.exists():
                raise ValueError(f"cascade artifacts missing: {artifact_dir}")
            cascade_artifacts = load_cascade_artifacts(artifact_dir)
            cascade_pipeline = CascadePipeline.from_artifacts(cascade_artifacts)

        return cls(
            model=model,
            soft_model=soft_model,
            pm1_model=pm1_model,
            feature_columns=cols,
            strategy=strategy,
            feature_version=metadata_feature_version,
            runtime_config=runtime_cfg,
            cascade_pipeline=cascade_pipeline,
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
            full_cand = build_candidate_matrix(
                row,
                V3_CORE20_COLUMNS,
                strict_features=False,
            )
            full_cand = full_cand.reset_index(drop=True)
            full_cand.insert(0, "number", np.arange(1, 81, dtype=int))
            x = None
            if not is_cascade_strategy(self.strategy):
                x = (
                    full_cand.reindex(columns=["number", *self.feature_columns])
                    .drop(columns=["number"])
                    .reindex(columns=self.feature_columns)
                )
        finally:
            if prev_runtime is None:
                os.environ.pop("FEATURE_RUNTIME_CONFIG_JSON", None)
            else:
                os.environ["FEATURE_RUNTIME_CONFIG_JSON"] = prev_runtime
            if prev_version is None:
                os.environ.pop("FEATURE_VERSION_OVERRIDE", None)
            else:
                os.environ["FEATURE_VERSION_OVERRIDE"] = prev_version
        regime = derive_regime(row)
        stage_debug = None
        if is_cascade_strategy(self.strategy):
            if self.cascade_pipeline is None:
                raise ValueError("cascade pipeline is not loaded")
            cascade = self.cascade_pipeline.predict_issue(row)
            scores = cascade["final_scores"]
            stage1_df = cascade["stage1"]
            stage2_df = cascade["stage2"]
            stage3_inputs = cascade["stage3_inputs"]
            stage2_top10 = (
                stage2_df.sort_values("stage2_score", ascending=False)["number"]
                .head(10)
                .astype(int)
                .tolist()
            )
            stage_debug = {
                "stage1_top5": stage1_df[["number", "stage1_score"]]
                .head(5)
                .to_dict(orient="records"),
                "stage1_keep_count": int(stage1_df["stage1_keep_flag"].sum()),
                "stage2_top5": stage2_df[["number", "stage2_score"]]
                .head(5)
                .to_dict(orient="records"),
                "stage2_keep_count": int(stage2_df["stage2_keep_flag"].sum()),
                "stage3_inputs_preview": stage3_inputs.head(5).to_dict(
                    orient="records"
                ),
                "selector": {
                    "final_top3": cascade.get("final_top3", []),
                    "no_selector_top3": cascade.get("no_selector_top3", []),
                    "selector_score": float(cascade.get("selector_score", 0.0)),
                    "selector_reason": cascade.get("selector_reason", ""),
                    "regime": cascade.get("selector_regime", "unknown"),
                },
            }
        else:
            if self.model is None or x is None:
                raise ValueError("legacy model not loaded")
            base_scores = self.model.predict_proba(x)[:, 1]
            scores = apply_strategy(base_scores, x, self.strategy, regime)

        soft_raw = None
        pm1_raw = None
        if x is not None and self.soft_model is not None:
            soft_raw = self.soft_model.predict(x)
        if x is not None and self.pm1_model is not None:
            pm1_raw = self.pm1_model.predict_proba(x)[:, 1]

        snapshot_payload = load_history_snapshot_payload()
        board_priors = snapshot_payload.get("meta", {}).get("board_priors", {})

        runtime_outputs = score_candidates_runtime(
            base_scores=np.array(scores, dtype=float),
            candidate_df=full_cand,
            recent_draws=[
                sorted(json.loads(v) if isinstance(v, str) else v)
                for v in draws_df["numbers"].tolist()
            ],
            runtime_config=self.runtime_config,
            snapshot_payload=snapshot_payload,
            board_priors=board_priors,
            soft_label_raw=soft_raw,
            pm1_proximity_raw=pm1_raw,
        )
        score_table = runtime_outputs.score_table
        rerank_summary = runtime_outputs.rerank_summary
        local_peak_summary = runtime_outputs.local_peak_summary
        dedup_summary = runtime_outputs.dedup_summary

        top20 = score_table["number"].head(20).astype(int).tolist()
        compact10 = compact_10_from_top20(top20)
        top10 = top20[:10]
        top3 = dedup_summary.get("top3_after_group_dedup", top20[:3])
        if is_cascade_strategy(self.strategy) and stage_debug is not None:
            sel = stage_debug.get("selector", {})
            picked = list(sel.get("final_top3", []))
            if len(picked) == 3:
                top3 = [int(x) for x in picked]
            if "stage2_top5" in stage_debug:
                top10 = stage2_top10
                if not bool(
                    self.runtime_config.get("topk_group_dedup", {}).get(
                        "enabled", False
                    )
                ):
                    top3 = top10[:3]
        latest_issue = int(row["issue"])
        zc = {z: sum(1 for n in top20 if zone_of(n) == z) for z in ["A", "B", "C", "D"]}
        board_type = classify_board(zc)
        candidate_cols = [
            "number",
            "model_score",
            "history_prior_score",
            "long_feature_score",
            "soft_label_score",
            "pm1_proximity_score",
            "score_before_analysis_rerank",
            "analysis_compatibility_score",
            "analysis_rerank_score",
            "score_after_analysis_rerank",
            "raw_score",
            "local_peak_score",
            "score_after_local_peak",
            "final_score",
            "cand_current_gap_all",
            "rank_model_only",
            "rank_final",
        ]
        score_table["score"] = score_table["final_score"].astype(float)
        raw_score_table = score_table[["score", *candidate_cols]].to_dict(
            orient="records"
        )
        score_table_compact = (
            score_table[
                [
                    "number",
                    "final_score",
                    "rank_final",
                    "model_score",
                    "score_after_local_peak",
                ]
            ]
            .rename(columns={"final_score": "score"})
            .to_dict(orient="records")
        )
        top20_scores = {
            f"{int(rec['number']):02d}": float(rec["final_score"])
            for rec in raw_score_table[:20]
        }
        big_count = sum(1 for n in top20 if n >= 41)
        odd_count = sum(1 for n in top20 if n % 2 == 1)
        model_rank_top20 = (
            score_table.sort_values("rank_model_only", ascending=True)["number"]
            .head(20)
            .astype(int)
            .tolist()
        )
        history_prior_summary = {
            "enabled": bool(
                self.runtime_config.get("history_prior", {}).get("enabled", True)
            ),
            "model_weight": float(
                self.runtime_config.get("history_prior", {}).get("model_weight", 0.88)
            ),
            "history_weight": float(
                self.runtime_config.get("history_prior", {}).get("history_weight", 0.12)
            ),
            "long_feature_enabled": bool(
                self.runtime_config.get("long_feature_injection", {}).get(
                    "enabled", True
                )
            ),
            "long_feature_weight": float(
                self.runtime_config.get("long_feature_injection", {}).get(
                    "weight", 0.06
                )
            ),
            "snapshot_status": snapshot_payload.get("status", "unavailable"),
            "snapshot_load_elapsed_ms": int(snapshot_payload.get("load_elapsed_ms", 0)),
            "top5_history_prior": score_table[
                ["number", "history_prior_score", "long_feature_score"]
            ]
            .head(5)
            .to_dict(orient="records"),
        }

        result = {
            "model": "catboost",
            "strategy_version": self.strategy.version_id,
            "target_issue": latest_issue + 1,
            "top20_numbers": top20,
            "top20_numbers_model": model_rank_top20,
            "top10_numbers": top10,
            "top10_stage2_ranked": top10,
            "top3_numbers": top3,
            "top3_no_selector": (
                stage_debug.get("selector", {}).get("no_selector_top3", top3)
                if stage_debug
                else top3
            ),
            "top3_selector_final": top3,
            "top20_scores": top20_scores,
            "compact10_numbers": compact10,
            "top3_core_group": top3,
            "raw_score_table": raw_score_table,
            "calibrated_probability_table": [
                {"number": x["number"], "probability": x["final_score"]}
                for x in raw_score_table
            ],
            "score_table": score_table_compact,
            "history_prior_score_summary": history_prior_summary,
            "analysis_rerank_summary": rerank_summary,
            "local_peak_summary": local_peak_summary,
            "grouped_candidates_preview": dedup_summary["grouped_candidates_preview"],
            "top3_before_group_dedup": dedup_summary["top3_before_group_dedup"],
            "top3_after_group_dedup": dedup_summary["top3_after_group_dedup"],
            "dedup_applied_scope": dedup_summary.get(
                "dedup_applied_scope", "top3_only"
            ),
            "final_score_breakdown": {
                "model_weight": float(
                    self.runtime_config.get("history_prior", {}).get(
                        "model_weight", 0.88
                    )
                ),
                "history_weight": float(
                    self.runtime_config.get("history_prior", {}).get(
                        "history_weight", 0.12
                    )
                ),
                "long_feature_weight": float(
                    self.runtime_config.get("long_feature_injection", {}).get(
                        "weight", 0.06
                    )
                ),
                "soft_label_weight": (
                    float(
                        self.runtime_config.get("soft_label_training", {}).get(
                            "blend_weight", 0.15
                        )
                    )
                    if bool(
                        self.runtime_config.get("soft_label_training", {}).get(
                            "enabled", False
                        )
                    )
                    and self.soft_model is not None
                    else 0.0
                ),
                "pm1_weight": (
                    float(
                        self.runtime_config.get("proximity_model", {}).get(
                            "pm1_weight", 0.12
                        )
                    )
                    if bool(
                        self.runtime_config.get("proximity_model", {}).get(
                            "enabled", False
                        )
                    )
                    and self.pm1_model is not None
                    else 0.0
                ),
                "analysis_rerank_weight": (
                    float(
                        self.runtime_config.get("analysis_rerank", {}).get(
                            "weight", 0.08
                        )
                    )
                    if bool(
                        self.runtime_config.get("analysis_rerank", {}).get(
                            "enabled", True
                        )
                    )
                    else 0.0
                ),
                "soft_label_normalization_method": str(
                    self.runtime_config.get("soft_label_training", {}).get(
                        "normalization", "rank_pct"
                    )
                ),
                "local_peak_enabled": bool(
                    self.runtime_config.get("neighbor_peak_correction", {}).get(
                        "enabled", False
                    )
                ),
                "group_dedup_enabled": bool(
                    self.runtime_config.get("topk_group_dedup", {}).get(
                        "enabled", False
                    )
                ),
            },
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
        if stage_debug is not None:
            result["cascade_debug"] = stage_debug
        return result


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "predict.yaml")
    normalize_pipeline_version(cfg.get("pipeline", {}).get("version"))
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

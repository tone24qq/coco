from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.strategy import (  # noqa: E402
    StrategyConfig,
    apply_strategy,
    default_experiments,
    derive_regime,
    issue_metrics,
    strategy_to_dict,
)
from src.utils import (  # noqa: E402
    CONFIG_DIR,
    FEATURE_STORE_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    load_yaml,
    precompute_issue_payloads,
    save_json,
    validate_feature_columns_contract,
)

METRIC_KEYS = [
    "top20_hit_rate",
    "top5_hit_rate",
    "top10_hit_rate",
    "top3_hit_rate",
    "top3_at_least_one_hit_rate",
    "ndcg_at_10",
]


def _expand_rows(
    issue_payloads: dict[int, dict[str, object]],
    indices: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    x_blocks, y_blocks = [], []
    for idx in indices:
        payload = issue_payloads[int(idx)]
        cand = payload["cand"]
        target = payload["target"]
        labels = pd.Series([1 if n in target else 0 for n in range(1, 81)])
        x_blocks.append(cand)
        y_blocks.append(labels)
    return pd.concat(x_blocks, ignore_index=True), pd.concat(
        y_blocks, ignore_index=True
    )


def _aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {k: 0.0 for k in METRIC_KEYS}
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in METRIC_KEYS}


def _overfit_audit(
    train_fold: list[dict], test_fold: list[dict], regime_rows: list[dict], th: dict
) -> dict:
    train_top3 = np.array([x["top3_hit_rate"] for x in train_fold], dtype=float)
    test_top3 = np.array([x["top3_hit_rate"] for x in test_fold], dtype=float)
    regime_df = pd.DataFrame(regime_rows)
    gap = float(train_top3.mean() - test_top3.mean()) if len(train_top3) else 0.0
    fold_disp = float(test_top3.std(ddof=0)) if len(test_top3) else 0.0
    regime_disp = (
        float(regime_df["top3_hit_rate"].std(ddof=0)) if not regime_df.empty else 0.0
    )
    overfit = bool(
        gap > float(th.get("train_vs_oos_gap", 0.03))
        or fold_disp > float(th.get("fold_dispersion", 0.05))
        or regime_disp > float(th.get("regime_dispersion", 0.06))
    )
    return {
        "train_vs_oos_gap": gap,
        "fold_dispersion": fold_disp,
        "regime_dispersion": regime_disp,
        "is_overfit": overfit,
    }


def _load_experiments() -> list[StrategyConfig]:
    exp_cfg_path = CONFIG_DIR / "experiments.yaml"
    if not exp_cfg_path.exists():
        return default_experiments()
    payload = load_yaml(exp_cfg_path)
    experiments = []
    for row in payload.get("experiments", []):
        experiments.append(StrategyConfig(**row))
    return experiments or default_experiments()


def _evaluate_strategies(
    feat_df: pd.DataFrame,
    issue_payloads: dict[int, dict[str, object]],
    params: dict,
    splits: int,
    experiments: list[StrategyConfig],
    overfit_th: dict,
) -> tuple[pd.DataFrame, dict, dict]:
    tss = TimeSeriesSplit(n_splits=splits)
    rows = []
    best = None
    baseline = None
    for exp in experiments:
        print(f"[版本開始] {exp.version_id}")
        fold_train, fold_test, regime_rows = [], [], []
        for fold_id, (tr_idx, te_idx) in enumerate(tss.split(feat_df), start=1):
            x_train, y_train = _expand_rows(issue_payloads, list(tr_idx))
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, verbose=False)

            def _score_issue(row_idx: int) -> tuple[dict, str]:
                payload = issue_payloads[int(row_idx)]
                cand = payload["cand"]
                base_scores = model.predict_proba(cand)[:, 1]
                regime = payload["regime"]
                if regime is None:
                    regime = derive_regime(feat_df.iloc[row_idx])
                    payload["regime"] = regime
                final_scores = apply_strategy(base_scores, cand, exp, regime)
                metric = issue_metrics(final_scores, payload["target"])
                return metric, regime

            train_eval_rows = []
            for row_idx in tr_idx[-min(50, len(tr_idx)) :]:
                m, _ = _score_issue(int(row_idx))
                train_eval_rows.append(m)
            test_eval_rows = []
            for row_idx in te_idx:
                m, regime = _score_issue(int(row_idx))
                test_eval_rows.append(m)
                regime_rows.append({"fold": fold_id, "regime": regime, **m})

            tr_agg = _aggregate(train_eval_rows)
            te_agg = _aggregate(test_eval_rows)
            fold_train.append({"fold": fold_id, **tr_agg})
            fold_test.append({"fold": fold_id, **te_agg})
            print(
                f"[Fold {fold_id}/{splits}] {exp.version_id} top20命中率={te_agg['top20_hit_rate']:.4f}"
            )
            print(
                f"[Fold {fold_id}/{splits}] {exp.version_id} top10命中率={te_agg['top10_hit_rate']:.4f}"
            )
            print(
                f"[Fold {fold_id}/{splits}] {exp.version_id} top3命中率={te_agg['top3_hit_rate']:.4f}"
            )
            print(
                f"[Fold {fold_id}/{splits}] {exp.version_id} top3至少中1顆率={te_agg['top3_at_least_one_hit_rate']:.4f}"
            )

        overall = _aggregate(fold_test)
        audit = _overfit_audit(fold_train, fold_test, regime_rows, overfit_th)
        if baseline is None:
            baseline = overall
        better = bool(
            overall["top3_at_least_one_hit_rate"]
            >= baseline["top3_at_least_one_hit_rate"]
            and overall["top3_hit_rate"] >= baseline["top3_hit_rate"]
        )
        keep = bool(better and not audit["is_overfit"])
        rec = {
            **strategy_to_dict(exp),
            **overall,
            **audit,
            "is_better_than_baseline": better,
            "keep_recommendation": keep,
        }
        rows.append(rec)
        if best is None or (
            rec["keep_recommendation"],
            rec["top3_at_least_one_hit_rate"],
            rec["top3_hit_rate"],
        ) > (
            best["keep_recommendation"],
            best["top3_at_least_one_hit_rate"],
            best["top3_hit_rate"],
        ):
            best = rec

    if best is None:
        raise ValueError("no strategy evaluated")
    if not bool(best["keep_recommendation"]):
        best = rows[0]
    return pd.DataFrame(rows), best, rows[0]


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    os.environ["STRICT_FEATURES"] = "1"
    feature_df = pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv")
    max_draws = int(cfg.get("max_draws_for_training", len(feature_df)))
    feature_df = feature_df.tail(max_draws).reset_index(drop=True)
    feature_columns = json.loads(
        (MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8")
    )
    feature_version = str(cfg.get("feature_version", "v3_core20"))
    if feature_version != "v3_core20":
        raise ValueError("only v3_core20 is supported")
    validate_feature_columns_contract(feature_columns, feature_version)

    if len(feature_df) < 3000:
        raise ValueError("訓練資料不足 3000 期，請先更新資料。")

    params = cfg.get("catboost_params", {})
    params.setdefault("loss_function", "Logloss")
    params.setdefault("verbose", False)
    params.setdefault("random_seed", 42)

    print("[訓練開始] 模型：CatBoost Binary")
    print(f"[訓練設定] max_draws_for_training={max_draws}")
    print(
        f"[資料摘要] 訓練期數：{len(feature_df)}，特徵數：{len(feature_columns)}，模型類型：catboost"
    )

    all_experiments = _load_experiments()
    fast_version_ids = {
        "v0_binary_baseline",
        "v3_rerank_k30_p300",
        "v4_two_stage_20_10_3",
    }
    fast_experiments = [
        exp for exp in all_experiments if exp.version_id in fast_version_ids
    ]
    issue_payloads = precompute_issue_payloads(
        feature_df,
        feature_columns,
        strict_features=True,
    )

    print("[研究流程] 快速階段：3個版本、3 folds、較低 iterations")
    fast_params = dict(params)
    fast_params["iterations"] = int(cfg.get("research_iterations", 140))
    fast_registry_df, fast_best, _ = _evaluate_strategies(
        feature_df,
        issue_payloads,
        params=fast_params,
        splits=int(cfg.get("research_backtest_splits", 3)),
        experiments=fast_experiments,
        overfit_th=cfg.get("overfit_thresholds", {}),
    )

    candidates = [
        row
        for row in fast_registry_df.sort_values(
            ["keep_recommendation", "top3_at_least_one_hit_rate", "top3_hit_rate"],
            ascending=False,
        ).to_dict(orient="records")
        if row["version_id"] in fast_version_ids
    ]
    selected_final_ids = [
        x["version_id"] for x in candidates[: int(cfg.get("final_stage_versions", 2))]
    ]
    if not selected_final_ids:
        selected_final_ids = [fast_best["version_id"]]
    if "v0_binary_baseline" not in selected_final_ids:
        selected_final_ids = ["v0_binary_baseline", *selected_final_ids]
    final_experiments = [
        exp for exp in all_experiments if exp.version_id in set(selected_final_ids)
    ]

    print(
        f"[研究流程] 正式階段：版本={selected_final_ids}、{int(cfg.get('backtest_splits', 5))} folds"
    )
    registry_df, best, baseline = _evaluate_strategies(
        feature_df,
        issue_payloads,
        params=params,
        splits=int(cfg.get("backtest_splits", 5)),
        experiments=final_experiments,
        overfit_th=cfg.get("overfit_thresholds", {}),
    )

    x_train, y_train = _expand_rows(issue_payloads, list(range(len(feature_df))))
    final_model = CatBoostClassifier(**params)
    final_model.fit(x_train, y_train, verbose=False)
    final_model.save_model(str(MODELS_DIR / "catboost_top20.cbm"))

    importances = final_model.get_feature_importance()
    fi_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": [float(x) for x in importances],
        }
    ).sort_values("importance", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry_df.to_csv(REPORTS_DIR / "experiment_registry.csv", index=False)
    fast_registry_df.to_csv(
        REPORTS_DIR / "experiment_registry_research.csv", index=False
    )
    fi_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    save_json(
        REPORTS_DIR / "feature_importance.json",
        {
            "features": fi_df.to_dict(orient="records"),
        },
    )

    print(
        "[整體結果] "
        f"top20_hit_rate={best['top20_hit_rate']:.4f}, "
        f"top10_hit_rate={best['top10_hit_rate']:.4f}, "
        f"top3_hit_rate={best['top3_hit_rate']:.4f}, "
        f"top3_at_least_one_hit_rate={best['top3_at_least_one_hit_rate']:.4f}"
    )
    print(
        "[過擬合檢查] "
        f"gap={best['train_vs_oos_gap']:.4f}, "
        f"fold_dispersion={best['fold_dispersion']:.4f}, "
        f"regime_dispersion={best['regime_dispersion']:.4f}, "
        f"overfit={bool(best['is_overfit'])}"
    )
    print(f"[最佳版本] {best['version_id']}")
    print(
        f"[正式預測版本] {best['version_id'] if bool(best.get('keep_recommendation')) else baseline['version_id']}"
    )

    strategy_payload = {
        "selected_strategy": best,
        "fallback_strategy": baseline,
    }
    save_json(MODELS_DIR / "strategy_config.json", strategy_payload)

    metadata = {
        "model_type": "catboost",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_rows": int(len(feature_df)),
        "feature_count": len(feature_columns),
        "train_issue_start": int(feature_df["issue"].min()),
        "train_issue_end": int(feature_df["target_issue"].max()),
        "feature_columns_path": "models/feature_columns.json",
        "model_path": "models/catboost_top20.cbm",
        "params": params,
        "selected_strategy": best,
        "fallback_strategy": baseline,
        "feature_version": feature_version,
        "runtime_config": {
            "core_windows": cfg.get("core_windows", {}),
            "smoothing_alpha": cfg.get("smoothing_alpha", 0.5),
            "decay_half_lives": cfg.get("decay_half_lives", {}),
            "distance_kernel_tau": cfg.get("distance_kernel_tau", 2),
        },
    }
    from src.utils import V3_CORE20_COLUMNS

    if len(feature_columns) != len(V3_CORE20_COLUMNS):
    raise ValueError(
        f"v3_core20 metadata requires feature_count={len(V3_CORE20_COLUMNS)}"
    )

save_json(MODELS_DIR / "metadata.json", metadata)

if __name__ == "__main__":
    main()

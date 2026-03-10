from __future__ import annotations

import json
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
    build_candidate_matrix,
    load_yaml,
    save_json,
)

METRIC_KEYS = [
    "top20_hit_rate",
    "top10_hit_rate",
    "top3_hit_rate",
    "top3_at_least_one_hit_rate",
]


def _expand_rows(
    feature_df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    x_blocks, y_blocks = [], []
    for _, row in feature_df.iterrows():
        cand = build_candidate_matrix(row, feature_columns)
        target = set(json.loads(row["target_numbers"]))
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
    feature_columns: list[str],
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
        fold_train, fold_test, regime_rows = [], [], []
        for fold_id, (tr_idx, te_idx) in enumerate(tss.split(feat_df), start=1):
            train_df = feat_df.iloc[tr_idx]
            test_df = feat_df.iloc[te_idx]
            x_train, y_train = _expand_rows(train_df, feature_columns)
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, verbose=False)

            def _score_issue(row: pd.Series) -> tuple[dict, str]:
                cand = build_candidate_matrix(row, feature_columns)
                base_scores = model.predict_proba(cand)[:, 1]
                regime = derive_regime(row)
                final_scores = apply_strategy(base_scores, cand, exp, regime)
                metric = issue_metrics(
                    final_scores, set(json.loads(row["target_numbers"]))
                )
                return metric, regime

            train_eval_rows = []
            for _, r in train_df.tail(min(50, len(train_df))).iterrows():
                m, _ = _score_issue(r)
                train_eval_rows.append(m)
            test_eval_rows = []
            for _, r in test_df.iterrows():
                m, regime = _score_issue(r)
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
    feature_df = pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv")
    feature_columns = json.loads(
        (MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8")
    )

    if len(feature_df) < 3000:
        raise ValueError("訓練資料不足 3000 期，請先更新資料。")

    params = cfg.get("catboost_params", {})
    params.setdefault("loss_function", "Logloss")
    params.setdefault("verbose", False)
    params.setdefault("random_seed", 42)

    print("[訓練開始] 模型：CatBoost Binary")
    print(
        f"[資料摘要] 訓練期數：{len(feature_df)}，特徵數：{len(feature_columns)}，模型類型：catboost"
    )

    experiments = _load_experiments()
    registry_df, best, baseline = _evaluate_strategies(
        feature_df,
        feature_columns,
        params=params,
        splits=int(cfg.get("backtest_splits", 5)),
        experiments=experiments,
        overfit_th=cfg.get("overfit_thresholds", {}),
    )

    x_train, y_train = _expand_rows(feature_df, feature_columns)
    final_model = CatBoostClassifier(**params)
    final_model.fit(x_train, y_train, verbose=False)
    final_model.save_model(str(MODELS_DIR / "catboost_top20.cbm"))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry_df.to_csv(REPORTS_DIR / "experiment_registry.csv", index=False)

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
        "feature_version": "v2",
    }
    save_json(MODELS_DIR / "metadata.json", metadata)


if __name__ == "__main__":
    main()

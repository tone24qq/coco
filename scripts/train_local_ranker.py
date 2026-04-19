from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.safe_io import SafeWriteConfig, read_dataset_auto, write_dataframe_safe
from src.whole_board_features import (
    FEATURE_SCHEMA_VERSION,
    NEAR_CONSTANT_DOMINANT_RATIO,
    NEAR_CONSTANT_STD_EPS,
    is_dynamic_optional_feature_column,
    is_primary_feature_column,
)


def _filter_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"group_id", "label", "is_feasible"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    out = df.copy()
    out = out[out["is_feasible"].astype(int) == 1].copy()
    label_sum = out.groupby("group_id")["label"].sum()
    out = out[out["group_id"].isin(label_sum[label_sum == 1].index)].copy()
    group_sizes = out.groupby("group_id").size()
    out = out[out["group_id"].isin(group_sizes[group_sizes >= 2].index)].copy()
    if out.empty:
        raise ValueError("no feasible ranking rows left after filtering")
    return out


def _near_constant_features(df: pd.DataFrame, cols: List[str]) -> List[str]:
    out: List[str] = []
    for col in cols:
        s = df[col].fillna(0.0)
        if s.nunique(dropna=False) <= 1:
            out.append(col)
            continue
        if float(s.std()) <= NEAR_CONSTANT_STD_EPS:
            out.append(col)
            continue
        dominant = float(s.value_counts(normalize=True, dropna=False).iloc[0])
        if dominant >= NEAR_CONSTANT_DOMINANT_RATIO:
            out.append(col)
    return sorted(set(out))


def _feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    all_candidate_cols = [
        c for c in df.columns if c.startswith("board_state_") or c.startswith("candidate_delta_")
    ]
    primary = [c for c in all_candidate_cols if is_primary_feature_column(c)]
    if not primary:
        raise ValueError("no primary residue/multiple10 feature columns")
    dead = _near_constant_features(df, primary)
    selected = [c for c in primary if c not in set(dead)]
    if not selected:
        raise ValueError("all primary features are near-constant; refuse to train")
    deprecated = [c for c in all_candidate_cols if c not in selected]
    return selected, deprecated, dead


def _validate_schema(df: pd.DataFrame, feature_columns: List[str], schema_path: str) -> None:
    if not schema_path:
        return
    path = Path(schema_path)
    if not path.exists():
        raise ValueError(f"feature schema file missing: {schema_path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    schema_features = list(schema.get("feature_columns", []))
    if not schema_features:
        raise ValueError("feature schema has empty feature_columns")

    missing_in_df = sorted(set(schema_features) - set(df.columns))
    hard_missing = [c for c in missing_in_df if not is_dynamic_optional_feature_column(c)]
    if hard_missing:
        raise ValueError(f"schema mismatch: dataset missing features {hard_missing[:8]}")

    missing_in_schema = sorted(set(feature_columns) - set(schema_features))
    if missing_in_schema:
        raise ValueError(f"schema mismatch: selected features absent in schema {missing_in_schema[:8]}")


def _metrics(df: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
    ranked = df[["group_id", "label"]].copy()
    ranked["score"] = scores

    top_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    ranks: List[int] = []
    for _, g in ranked.groupby("group_id", sort=False):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        pos = g.index[g["label"] == 1].tolist()
        if not pos:
            continue
        rk = int(pos[0] + 1)
        ranks.append(rk)
        for k in top_hits:
            top_hits[k] += int(rk <= k)

    total = max(len(ranks), 1)
    cand_mean = float(df.groupby("group_id").size().mean()) if total > 0 else 0.0
    mrr = float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0
    mean_rank = float(np.mean(ranks)) if ranks else 0.0
    norm_gain = 1.0
    if cand_mean > 1.0:
        norm_gain = float(1.0 - (mean_rank - 1.0) / (cand_mean - 1.0))
    return {
        "group_count": int(total),
        "top1": top_hits[1] / total,
        "top3": top_hits[3] / total,
        "top5": top_hits[5] / total,
        "top10": top_hits[10] / total,
        "mean_rank": mean_rank,
        "mrr": mrr,
        "candidate_count_mean": cand_mean,
        "normalized_mean_rank_gain": norm_gain,
    }


def _train(backend: str, params: Dict[str, Any], x: np.ndarray, y: np.ndarray, groups: List[int], n_jobs: int):
    if backend == "lightgbm":
        from lightgbm import LGBMRanker  # type: ignore

        model = LGBMRanker(objective="lambdarank", n_jobs=n_jobs, **params)
        model.fit(x, y, group=groups)
        return model

    model = HistGradientBoostingClassifier(**params)
    model.fit(x, y)
    return model


def _size_effective_counts(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if "size_class" not in df.columns:
        return out
    for size, sub in df.groupby("size_class"):
        valid = []
        for col in cols:
            s = sub[col].fillna(0.0) if col in sub.columns else pd.Series([], dtype=float)
            if len(s) > 0 and s.nunique(dropna=False) > 1 and float(s.std()) > NEAR_CONSTANT_STD_EPS:
                valid.append(col)
        out[str(size)] = len(valid)
    return out


def train_once(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    backend: str,
    params: Dict[str, Any],
    n_jobs: int,
    feature_schema_path: str = "",
) -> Tuple[Any, List[str], Dict[str, Any]]:
    train_df = _filter_training_rows(train_df).sort_values(["group_id", "cand_row", "cand_col"]).reset_index(drop=True)
    valid_df = _filter_training_rows(valid_df).sort_values(["group_id", "cand_row", "cand_col"]).reset_index(drop=True)
    holdout_df = _filter_training_rows(holdout_df).sort_values(["group_id", "cand_row", "cand_col"]).reset_index(drop=True)

    features, deprecated, dead = _feature_columns(train_df)
    _validate_schema(train_df, features, feature_schema_path)

    x_train = train_df[features].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_df["label"].astype(int).to_numpy()
    groups = train_df.groupby("group_id", sort=False).size().tolist()

    model = _train(backend, params, x_train, y_train, groups, n_jobs)

    def _pred(df: pd.DataFrame) -> np.ndarray:
        x = df[features].fillna(0.0).to_numpy(dtype=np.float32)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1]
        return model.predict(x)

    metrics = {
        "valid": _metrics(valid_df, _pred(valid_df)),
        "holdout": _metrics(holdout_df, _pred(holdout_df)),
    }
    return model, features, {
        "dead_or_near_constant_features": dead,
        "deprecated_features": deprecated,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_group_count": int(valid_df["group_id"].nunique()),
        "holdout_group_count": int(holdout_df["group_id"].nunique()),
        "size_effective_feature_count": _size_effective_counts(train_df, features),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--holdout-path", required=True)
    parser.add_argument("--backend", choices=["lightgbm", "sklearn"], default="lightgbm")
    parser.add_argument("--params-json", default="")
    parser.add_argument("--params-path", default="")
    parser.add_argument("--size-class", default="")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--artifacts-dir", default="artifacts/global")
    parser.add_argument("--report", default="reports/train_local_ranker_report.json")
    parser.add_argument("--feature-schema", default="artifacts/feature_schema_residue.json")
    parser.add_argument("--max-file-mb", type=int, default=100)
    args = parser.parse_args()
    print("[訓練進度 10%] 已讀取訓練參數，準備載入資料。")

    if args.params_path:
        params = json.loads(Path(args.params_path).read_text(encoding="utf-8"))
    elif args.params_json:
        params = json.loads(args.params_json)
    else:
        if args.backend == "lightgbm":
            params = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
            }
        else:
            params = {"max_depth": 8, "learning_rate": 0.06, "max_iter": 300}

    train_df = read_dataset_auto(Path(args.train_path))
    valid_df = read_dataset_auto(Path(args.valid_path))
    holdout_df = read_dataset_auto(Path(args.holdout_path))
    print("[訓練進度 30%] 資料載入完成，準備篩選與模型訓練。")

    if args.size_class:
        train_df = train_df[train_df["size_class"] == args.size_class].copy()
        valid_df = valid_df[valid_df["size_class"] == args.size_class].copy()
        holdout_df = holdout_df[holdout_df["size_class"] == args.size_class].copy()
        print(f"[訓練進度 40%] 已套用 size_class={args.size_class} 篩選。")
        if train_df.empty or valid_df.empty or holdout_df.empty:
            raise ValueError(f"insufficient rows after size filter: size_class={args.size_class}")

    model, feature_columns, run = train_once(
        train_df,
        valid_df,
        holdout_df,
        args.backend,
        params,
        args.max_workers,
        feature_schema_path=args.feature_schema,
    )
    print("[訓練進度 70%] 模型訓練完成，準備寫出 artifacts 與報告。")

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "main_ranker.pkl"
    joblib.dump({"model": model, "feature_columns": feature_columns, "backend": args.backend}, model_path)

    meta = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "new_primary_feature_count": len(feature_columns),
        "dead_or_near_constant_features": run.get("dead_or_near_constant_features", []),
        "deprecated_features": run.get("deprecated_features", []),
        "backend": args.backend,
        "feature_columns": feature_columns,
        "size_class": args.size_class or "global",
        "params": params,
        "train_rows": run["train_rows"],
        "valid_rows": run["valid_rows"],
        "holdout_rows": run["holdout_rows"],
        "holdout_metrics": run["metrics"]["holdout"],
        "feasible_only_training": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (artifacts_dir / "main_ranker_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "new_primary_feature_count": len(feature_columns),
        "dead_or_near_constant_features": run.get("dead_or_near_constant_features", []),
        "deprecated_features": run.get("deprecated_features", []),
        "size_effective_feature_count": run.get("size_effective_feature_count", {}),
        "backend_used": args.backend,
        "size_class": args.size_class or "global",
        "params": params,
        **run,
        "feature_columns": feature_columns,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    trials_df = pd.DataFrame(
        [
            {
                "trial_id": 0,
                "backend": args.backend,
                "params_json": json.dumps(params, ensure_ascii=False),
                "train_rows": run["train_rows"],
                "valid_rows": run["valid_rows"],
                "valid_group_count": run["valid_group_count"],
                **run["metrics"]["valid"],
                "objective_score": run["metrics"]["valid"]["top1"],
                "elapsed_sec": 0.0,
                "model_size_mb": model_path.stat().st_size / (1024 * 1024),
            }
        ]
    )
    write_dataframe_safe(
        trials_df,
        report_path.with_name("train_local_ranker_trials.csv"),
        fmt="csv",
        config=SafeWriteConfig(max_file_mb=args.max_file_mb, producer_script="scripts/train_local_ranker.py"),
    )

    print("[訓練進度 100%] 訓練與報告輸出完成。")
    print(json.dumps({"status": "ok", "artifact": str(model_path), "metrics": run["metrics"]["holdout"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

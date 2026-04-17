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


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("board_state_") or c.startswith("candidate_delta_")]
    if not cols:
        raise ValueError("no feature columns")
    return cols


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


def train_once(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    backend: str,
    params: Dict[str, Any],
    n_jobs: int,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    train_df = _filter_training_rows(train_df).sort_values(["group_id", "cand_row", "cand_col"]).reset_index(drop=True)
    valid_df = _filter_training_rows(valid_df).sort_values(["group_id", "cand_row", "cand_col"]).reset_index(drop=True)
    holdout_df = _filter_training_rows(holdout_df).sort_values(["group_id", "cand_row", "cand_col"]).reset_index(drop=True)

    features = _feature_columns(train_df)
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
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_group_count": int(valid_df["group_id"].nunique()),
        "holdout_group_count": int(holdout_df["group_id"].nunique()),
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
    parser.add_argument("--max-file-mb", type=int, default=100)
    args = parser.parse_args()

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

    if args.size_class:
        train_df = train_df[train_df["size_class"] == args.size_class].copy()
        valid_df = valid_df[valid_df["size_class"] == args.size_class].copy()
        holdout_df = holdout_df[holdout_df["size_class"] == args.size_class].copy()

    model, feature_columns, run = train_once(train_df, valid_df, holdout_df, args.backend, params, args.max_workers)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "main_ranker.pkl"
    joblib.dump({"model": model, "feature_columns": feature_columns, "backend": args.backend}, model_path)

    meta = {
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
        "backend_used": args.backend,
        "size_class": args.size_class or "global",
        "params": params,
        **run,
        "feature_columns": feature_columns,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # aligned tabular report
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

    print(json.dumps({"status": "ok", "artifact": str(model_path), "metrics": run["metrics"]["holdout"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

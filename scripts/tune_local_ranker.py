from __future__ import annotations

import argparse
import itertools
import json
import random
import signal
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_local_ranker import train_once
from src.safe_io import SafeWriteConfig, read_dataset_auto, write_dataframe_safe

INTERRUPTED = False


def _handle_interrupt(_signum, _frame):
    global INTERRUPTED
    INTERRUPTED = True


def _space_lightgbm() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 200, 300, 500, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [5, 10, 20, 30, 50],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0, 2.0],
        "reg_lambda": [0, 0.1, 0.5, 1.0, 2.0, 5.0],
        "max_depth": [-1, 4, 6, 8, 10],
        "min_split_gain": [0, 0.01, 0.03, 0.1],
    }


def _space_sklearn() -> Dict[str, List[Any]]:
    return {
        "max_depth": [3, 5, 8, 12, None],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "max_iter": [100, 200, 300, 500],
        "min_samples_leaf": [10, 20, 30, 50],
    }


def _iter_grid(space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(space.keys())
    for vals in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, vals))


def _iter_random(space: Dict[str, List[Any]], n_trials: int, rng: random.Random) -> Iterable[Dict[str, Any]]:
    keys = list(space.keys())
    for _ in range(n_trials):
        yield {k: rng.choice(space[k]) for k in keys}


def _objective(metrics: Dict[str, float], profile: str) -> float:
    t1, t3, t5, t10 = metrics["top1"], metrics["top3"], metrics["top5"], metrics["top10"]
    mrr, nmg = metrics["mrr"], metrics.get("normalized_mean_rank_gain", 0.0)
    if profile == "hit_focus":
        return 0.40 * t1 + 0.30 * t3 + 0.20 * t5 + 0.10 * t10
    if profile == "rank_focus":
        return 0.20 * t1 + 0.20 * t3 + 0.20 * t10 + 0.20 * mrr + 0.20 * nmg
    return 0.30 * t1 + 0.25 * t3 + 0.15 * t5 + 0.10 * t10 + 0.10 * mrr + 0.10 * nmg


def main() -> None:
    signal.signal(signal.SIGINT, _handle_interrupt)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--holdout-path", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts/global")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--backend", choices=["lightgbm", "sklearn", "auto"], default="auto")
    parser.add_argument("--search-method", choices=["grid", "random", "optuna"], default="random")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--size-class", default="")
    parser.add_argument("--metric-profile", choices=["hit_focus", "balanced", "rank_focus"], default="balanced")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-file-mb", type=int, default=100)
    parser.add_argument("--trial-timeout-sec", type=int, default=0)
    parser.add_argument("--feature-schema", default="")
    args = parser.parse_args()
    print("[調參進度 5%] 參數解析完成，準備初始化調參流程。")

    backend = args.backend
    if backend == "auto":
        try:
            import lightgbm  # noqa: F401

            backend = "lightgbm"
        except Exception:
            backend = "sklearn"

    train_df = read_dataset_auto(Path(args.train_path))
    valid_df = read_dataset_auto(Path(args.valid_path))
    holdout_df = read_dataset_auto(Path(args.holdout_path))
    print("[調參進度 15%] 訓練/驗證/測試資料載入完成。")

    if args.size_class:
        train_df = train_df[train_df["size_class"] == args.size_class].copy()
        valid_df = valid_df[valid_df["size_class"] == args.size_class].copy()
        holdout_df = holdout_df[holdout_df["size_class"] == args.size_class].copy()
        print(f"[調參進度 20%] 已套用 size_class={args.size_class}。")

    valid_real_df = valid_df[valid_df["source_type"] == "real"].copy() if "source_type" in valid_df.columns else valid_df.copy()
    holdout_real_df = holdout_df[holdout_df["source_type"] == "real"].copy() if "source_type" in holdout_df.columns else holdout_df.copy()
    if valid_real_df.empty:
        raise ValueError("valid split has no real groups")
    if holdout_real_df.empty:
        raise ValueError("holdout split has no real groups")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = report_dir / "tuning_trials.csv"

    trial_rows: List[Dict[str, Any]] = []
    done_signatures = set()
    if args.resume and trials_csv.exists():
        prev = pd.read_csv(trials_csv)
        trial_rows = prev.to_dict(orient="records")
        done_signatures = {str(x) for x in prev["params_json"].tolist()}

    space = _space_lightgbm() if backend == "lightgbm" else _space_sklearn()
    rng = random.Random(args.seed)
    if args.search_method == "grid":
        iterator = _iter_grid(space)
    elif args.search_method in {"random", "optuna"}:
        iterator = _iter_random(space, args.n_trials * 3, rng)
    else:
        raise ValueError("unsupported search method")

    best: Dict[str, Any] | None = None
    print("[調參進度 30%] 開始 trial 搜尋。")
    for i, params in enumerate(iterator, start=1):
        if INTERRUPTED:
            break
        params_sig = json.dumps(params, sort_keys=True)
        if params_sig in done_signatures:
            continue
        t0 = time.time()
        try:
            _, _, run = train_once(
                train_df, valid_real_df, holdout_real_df, backend, params, args.max_workers, args.feature_schema
            )
        except Exception:
            if args.strict:
                raise
            continue
        elapsed = max(time.time() - t0, 1e-9)
        if args.trial_timeout_sec > 0 and elapsed > args.trial_timeout_sec:
            if args.strict:
                raise TimeoutError(f"trial exceeded timeout: {elapsed:.2f}s")
            continue
        valid_metrics = run["metrics"]["valid"]
        score = _objective(valid_metrics, args.metric_profile)
        row = {
            "trial_id": len(trial_rows),
            "backend": backend,
            "params_json": params_sig,
            "train_rows": run["train_rows"],
            "valid_rows": run["valid_rows"],
            "valid_group_count": run["valid_group_count"],
            "top1": valid_metrics["top1"],
            "top3": valid_metrics["top3"],
            "top5": valid_metrics["top5"],
            "top10": valid_metrics["top10"],
            "mean_rank": valid_metrics["mean_rank"],
            "mrr": valid_metrics["mrr"],
            "objective_score": score,
            "elapsed_sec": elapsed,
            "model_size_mb": 0.0,
        }
        trial_rows.append(row)
        done_signatures.add(params_sig)
        pct = min(90, int(30 + 60 * (len(trial_rows) / max(args.n_trials, 1))))
        print(f"[調參進度 {pct}%] trial {len(trial_rows)}/{args.n_trials} score={score:.4f} params={params_sig}")
        if best is None or row["objective_score"] > best["objective_score"]:
            best = row
        if len(trial_rows) >= args.n_trials:
            break

    if not trial_rows:
        raise ValueError("no tuning trials executed")

    trials_df = pd.DataFrame(trial_rows).sort_values("objective_score", ascending=False).reset_index(drop=True)
    best = trials_df.iloc[0].to_dict()
    best_params = json.loads(str(best["params_json"]))
    print("[調參進度 92%] trial 搜尋完成，開始最佳參數重訓。")

    # retrain on train+valid and evaluate once on holdout
    valid_for_train = valid_df.copy()
    valid_for_train["group_id"] = valid_for_train["group_id"].map(lambda x: f"valid::{x}")
    train_valid = pd.concat([train_df, valid_for_train], ignore_index=True)
    model, feature_columns, final_run = train_once(
        train_valid,
        valid_real_df,
        holdout_real_df,
        backend,
        best_params,
        args.max_workers,
        args.feature_schema,
    )

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    model_path = artifacts_dir / "main_ranker.pkl"
    joblib.dump({"model": model, "feature_columns": feature_columns, "backend": backend}, model_path)
    meta = {
        "size_class": args.size_class or "global",
        "backend": backend,
        "feature_schema_version": "whole_board_features_v3_refactored",
        "new_primary_feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "params": best_params,
        "train_rows": int(len(train_valid)),
        "valid_rows": int(len(valid_df)),
        "holdout_rows": int(len(holdout_df)),
        "holdout_metrics": final_run["metrics"]["holdout"],
    }
    (artifacts_dir / "main_ranker_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifacts_dir / "best_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    write_dataframe_safe(
        trials_df,
        trials_csv,
        fmt="csv",
        config=SafeWriteConfig(max_file_mb=args.max_file_mb, producer_script="scripts/tune_local_ranker.py"),
    )

    leaderboard = trials_df.head(10).to_dict(orient="records")
    (report_dir / "tuning_leaderboard.json").write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "search_method": args.search_method,
        "n_trials": int(len(trials_df)),
        "metric_profile": args.metric_profile,
        "best_trial_id": int(best["trial_id"]),
        "best_params": best_params,
        "best_valid_metrics": {
            "top1": best["top1"],
            "top3": best["top3"],
            "top5": best["top5"],
            "top10": best["top10"],
            "mean_rank": best["mean_rank"],
            "mrr": best["mrr"],
            "objective_score": best["objective_score"],
        },
        "best_valid_real_metrics": {
            "top1": best["top1"],
            "top3": best["top3"],
            "top5": best["top5"],
            "top10": best["top10"],
            "mean_rank": best["mean_rank"],
            "mrr": best["mrr"],
            "objective_score": best["objective_score"],
        },
        "final_holdout_metrics": final_run["metrics"]["holdout"],
        "final_holdout_real_metrics": final_run["metrics"]["holdout"],
        "valid_contains_synth": bool(("source_type" in valid_df.columns) and (valid_df["source_type"] == "synthetic").any()),
        "holdout_contains_synth": bool(("source_type" in holdout_df.columns) and (holdout_df["source_type"] == "synthetic").any()),
        "size_class": args.size_class or "global",
        "backend_used": backend,
        "new_primary_feature_count": len(feature_columns),
        "interrupted": INTERRUPTED,
    }
    (report_dir / "tuning_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[調參進度 100%] 調參與報告輸出完成。")


if __name__ == "__main__":
    main()

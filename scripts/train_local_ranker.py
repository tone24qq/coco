from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.hardware_profile import choose_training_plan, detect_hardware_profile, to_dict


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.is_file() and path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.is_dir() and (path / "manifest.json").exists():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

        if "files" in manifest:
            files = manifest.get("files", [])
            if not files:
                raise ValueError(f"no shard files in {path}")
            return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

        if "shards" in manifest:
            shards = manifest.get("shards", [])
            if not shards:
                raise ValueError(f"no shard records in {path}")
            return pd.concat(
                [pd.read_parquet(item["path"]) for item in shards],
                ignore_index=True,
            )

    raise ValueError(f"unsupported dataset path: {path}")


def _filter_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"group_id", "label", "is_feasible"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    out = df.copy()
    out = out[out["is_feasible"].astype(int) == 1].copy()

    # 每個 group 必須只剩一個正例，否則丟掉
    label_sum = out.groupby("group_id")["label"].sum()
    valid_groups = label_sum[label_sum == 1].index
    out = out[out["group_id"].isin(valid_groups)].copy()

    # 至少要有 2 個候選，否則沒排名意義
    group_sizes = out.groupby("group_id").size()
    valid_groups = group_sizes[group_sizes >= 2].index
    out = out[out["group_id"].isin(valid_groups)].copy()

    if out.empty:
        raise ValueError("no feasible ranking rows left after filtering")

    return out


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        c
        for c in df.columns
        if c.startswith("board_state_") or c.startswith("candidate_delta_")
    ]
    if not cols:
        raise ValueError("no whole-board feature columns found")
    return cols


def _group_metrics(df: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
    ranked = df[["group_id", "label"]].copy()
    ranked["score"] = scores

    top_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_values: List[float] = []
    rank_values: List[int] = []

    for _, g in ranked.groupby("group_id", sort=False):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        pos = g.index[g["label"] == 1].tolist()
        if not pos:
            continue
        rk = int(pos[0] + 1)
        rank_values.append(rk)
        mrr_values.append(1.0 / rk)
        for k in top_hits:
            top_hits[k] += int(rk <= k)

    total = max(len(rank_values), 1)
    return {
        "top1": top_hits[1] / total,
        "top3": top_hits[3] / total,
        "top5": top_hits[5] / total,
        "top10": top_hits[10] / total,
        "mean_rank": float(np.mean(rank_values) if rank_values else 0.0),
        "mrr": float(np.mean(mrr_values) if mrr_values else 0.0),
    }


def _train_with_backend(plan_backend: str, x: np.ndarray, y: np.ndarray, group_sizes: List[int], n_jobs: int) -> Tuple[Any, str]:
    if plan_backend == "lightgbm":
        try:
            from lightgbm import LGBMRanker  # type: ignore

            model = LGBMRanker(
                objective="lambdarank",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                n_jobs=n_jobs,
            )
            model.fit(x, y, group=group_sizes)
            return model, "lightgbm_lambdarank"
        except Exception:
            pass

    model = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.06,
        max_iter=300,
    )
    model.fit(x, y)
    return model, "sklearn_hgb_classifier"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-real-path", required=True)
    parser.add_argument("--train-synth-path", default="")
    parser.add_argument("--holdout-real-path", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-workers", default="auto")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--report", default="reports/train_local_ranker_report.json")
    args = parser.parse_args()

    profile = detect_hardware_profile()
    plan = choose_training_plan(profile, requested_device=args.device, max_workers=args.max_workers)

    train_real = _read_dataset(Path(args.train_real_path))
    frames = [train_real]

    if args.train_synth_path:
        synth_path = Path(args.train_synth_path)
        if synth_path.exists():
            frames.append(_read_dataset(synth_path))

    train_df = pd.concat(frames, ignore_index=True)
    holdout_df = _read_dataset(Path(args.holdout_real_path))

    train_df = _filter_training_rows(train_df)
    holdout_df = _filter_training_rows(holdout_df)

    # LightGBMRanker 要求 group 連續
    train_df = train_df.sort_values(["group_id", "cand_row", "cand_col"]).reset_index(drop=True)
    holdout_df = holdout_df.sort_values(["group_id", "cand_row", "cand_col"]).reset_index(drop=True)

    feature_cols = _feature_columns(train_df)

    x_train = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_df["label"].astype(int).to_numpy()

    x_holdout = holdout_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_holdout = holdout_df["label"].astype(int).to_numpy()

    group_sizes = train_df.groupby("group_id", sort=False).size().tolist()

    start = time.time()
    model, backend = _train_with_backend(plan.backend, x_train, y_train, group_sizes, plan.n_jobs)

    if hasattr(model, "predict_proba"):
        hold_scores = model.predict_proba(x_holdout)[:, 1]
    else:
        hold_scores = model.predict(x_holdout)

    elapsed = max(time.time() - start, 1e-6)
    samples_per_sec = len(train_df) / elapsed

    report = {
        "hardware_profile": to_dict(profile),
        "training_plan": to_dict(plan),
        "backend_used": backend,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "train_group_count": int(train_df["group_id"].nunique()),
        "holdout_group_count": int(holdout_df["group_id"].nunique()),
        "per_size_counts_train": train_df["size_class"].value_counts().to_dict(),
        "per_size_counts_holdout": holdout_df["size_class"].value_counts().to_dict(),
        "elapsed_seconds": float(elapsed),
        "samples_per_sec": float(samples_per_sec),
        "holdout_auc": float(roc_auc_score(y_holdout, hold_scores)) if len(np.unique(y_holdout)) > 1 else None,
        "holdout_ranking": _group_metrics(holdout_df, hold_scores),
        "filtering": {
            "feasible_only": True,
            "require_single_positive_per_group": True,
            "min_group_size": 2,
        },
    }

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "backend": backend,
        },
        artifacts_dir / "main_ranker.pkl",
    )

    (artifacts_dir / "main_ranker_meta.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "backend": backend,
                "feature_columns": feature_cols,
                "artifact": "artifacts/main_ranker.pkl",
                "schema_path": "artifacts/feature_schema.json",
                "strict_missing_artifact": True,
                "feasible_only_training": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "backend": backend,
                "rows": len(train_df),
                "groups": int(train_df["group_id"].nunique()),
                "samples_per_sec": samples_per_sec,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
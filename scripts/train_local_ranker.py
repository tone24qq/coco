from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.hardware_profile import choose_training_plan, detect_hardware_profile, to_dict


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.is_dir() and (path / "manifest.json").exists():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        files = manifest.get("files", [])
        if not files:
            raise ValueError(f"no shard files in {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raise ValueError(f"unsupported dataset path: {path}")


def _group_metrics(df: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
    ranked = df[["group_id", "label"]].copy()
    ranked["score"] = scores
    top_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr = []
    ranks = []
    for _, g in ranked.groupby("group_id", sort=False):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        pos = g.index[g["label"] == 1].tolist()
        if not pos:
            continue
        rk = int(pos[0] + 1)
        ranks.append(rk)
        mrr.append(1.0 / rk)
        for k in top_hits:
            top_hits[k] += int(rk <= k)
    total = max(len(ranks), 1)
    return {
        "top1": top_hits[1] / total,
        "top3": top_hits[3] / total,
        "top5": top_hits[5] / total,
        "top10": top_hits[10] / total,
        "mean_rank": float(np.mean(ranks) if ranks else 0.0),
        "mrr": float(np.mean(mrr) if mrr else 0.0),
    }


def _train_with_backend(plan_backend: str, X: np.ndarray, y: np.ndarray, n_jobs: int) -> Tuple[Any, str]:
    if plan_backend == "lightgbm":
        try:
            from lightgbm import LGBMRanker  # type: ignore

            model = LGBMRanker(
                objective="lambdarank",
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                n_jobs=n_jobs,
            )
            return model, "lightgbm_lambdarank"
        except Exception:
            pass
    model = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.08, max_iter=300)
    model.fit(X, y)
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

    feature_cols = [
        c for c in train_df.columns if c.startswith("board_state_") or c.startswith("candidate_delta_")
    ]
    if not feature_cols:
        raise ValueError("no whole-board feature columns found")

    X_train = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_df["label"].astype(int).to_numpy()
    X_holdout = holdout_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_holdout = holdout_df["label"].astype(int).to_numpy()

    start = time.time()
    model, backend = _train_with_backend(plan.backend, X_train, y_train, plan.n_jobs)
    if backend == "lightgbm_lambdarank":
        group_sizes = train_df.groupby("group_id").size().tolist()
        model.fit(X_train, y_train, group=group_sizes)

    if hasattr(model, "predict_proba"):
        hold_scores = model.predict_proba(X_holdout)[:, 1]
    else:
        hold_scores = model.predict(X_holdout)

    elapsed = max(time.time() - start, 1e-6)
    samples_per_sec = len(train_df) / elapsed

    report = {
        "hardware_profile": to_dict(profile),
        "training_plan": to_dict(plan),
        "backend_used": backend,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "per_size_counts_train": train_df["size_class"].value_counts().to_dict(),
        "per_size_counts_holdout": holdout_df["size_class"].value_counts().to_dict(),
        "eta_seconds": float(elapsed),
        "samples_per_sec": float(samples_per_sec),
        "holdout_auc": float(roc_auc_score(y_holdout, hold_scores)) if len(np.unique(y_holdout)) > 1 else None,
        "holdout_ranking": _group_metrics(holdout_df, hold_scores),
    }

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": feature_cols, "backend": backend},
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
            {"backend": backend, "rows": len(train_df), "samples_per_sec": samples_per_sec},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

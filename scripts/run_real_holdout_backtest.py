from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.whole_board_features import is_primary_feature_column


def _read(path: Path) -> pd.DataFrame:
    if path.is_dir() and (path / "manifest.json").exists():
        files = json.loads((path / "manifest.json").read_text(encoding="utf-8")).get("files", [])
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)


def _metrics(df: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
    w = df[["group_id", "label", "size_class", "source_type"]].copy()
    w["score"] = scores
    per_size: Dict[str, Dict[str, float]] = {}

    def _calc(sub: pd.DataFrame) -> Dict[str, float]:
        ranks = []
        for _, g in sub.groupby("group_id", sort=False):
            g = g.sort_values("score", ascending=False).reset_index(drop=True)
            idx = g.index[g["label"] == 1].tolist()
            if idx:
                ranks.append(idx[0] + 1)
        if not ranks:
            return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0, "mean_rank": 0.0, "mrr": 0.0}
        arr = np.array(ranks)
        return {
            "top1": float(np.mean(arr <= 1)),
            "top3": float(np.mean(arr <= 3)),
            "top5": float(np.mean(arr <= 5)),
            "top10": float(np.mean(arr <= 10)),
            "mean_rank": float(np.mean(arr)),
            "mrr": float(np.mean(1.0 / arr)),
        }

    overall = _calc(w)
    for size, sub in w.groupby("size_class"):
        per_size[str(size)] = _calc(sub)

    return {"overall": overall, "per_size": per_size}


def _train_score(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> Dict[str, object]:
    feats = [
        c
        for c in train_df.columns
        if (c.startswith("board_state_") or c.startswith("candidate_delta_")) and is_primary_feature_column(c)
    ]
    if not feats:
        raise ValueError("no primary residue/multiple10 features found")
    model = HistGradientBoostingClassifier(max_depth=8, max_iter=250, learning_rate=0.08)
    model.fit(train_df[feats].fillna(0.0), train_df["label"].astype(int))
    scores = model.predict_proba(holdout_df[feats].fillna(0.0))[:, 1]
    return _metrics(holdout_df, scores)


def _assert_lineage_disjoint(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> None:
    overlap = set(train_df["lineage_id"].unique()) & set(holdout_df["lineage_id"].unique())
    if overlap:
        raise ValueError(f"lineage overlap detected: {sorted(list(overlap))[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-real", required=True)
    parser.add_argument("--train-synth", required=True)
    parser.add_argument("--holdout-real", required=True)
    parser.add_argument("--output", default="reports/real_holdout_backtest_summary.json")
    args = parser.parse_args()

    train_real = _read(Path(args.train_real))
    train_synth = _read(Path(args.train_synth))
    holdout = _read(Path(args.holdout_real))

    _assert_lineage_disjoint(train_real, holdout)

    summary = {
        "real_only_train": _train_score(train_real, holdout),
        "synth_only_train": _train_score(train_synth, holdout),
        "real_plus_synth_train": _train_score(pd.concat([train_real, train_synth], ignore_index=True), holdout),
        "ablation": {
            "real_vs_synth": "compare real_only_train.overall and synth_only_train.overall",
            "real_vs_combined": "compare real_only_train.overall and real_plus_synth_train.overall",
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

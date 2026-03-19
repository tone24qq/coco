from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score

from src.runtime_scoring import RuntimeWeights, score_candidates
from src.utils import DataContractError

REQUIRED_COLUMNS = {"issue", "candidate_number", "label", "group_id"}
NON_FEATURE_COLUMNS = {
    "issue",
    "draw_date",
    "candidate_number",
    "label",
    "group_id",
    "current_day_recent_10_profile",
    "current_day_recent_20_profile",
    "current_day_recent_n_profile",
}


@dataclass
class FoldResult:
    fold_id: int
    val_scored: pd.DataFrame
    train_scored: pd.DataFrame
    train_issues: list[str]
    val_issues: list[str]


def load_ranking_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataContractError(f"ranking dataset not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataContractError(f"ranking dataset missing required columns: {sorted(missing)}")
    if df.empty:
        raise DataContractError("ranking dataset is empty")
    bad = df.groupby("issue")["candidate_number"].count()
    if (bad != 80).any():
        raise DataContractError("group ranking contract violated: each issue must have 80 candidates")
    return df


def resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    if not cols:
        raise DataContractError("no feature columns resolved")
    return cols


def _groups(df: pd.DataFrame) -> list[int]:
    return [len(g) for _, g in df.groupby("issue", sort=False)]


def make_time_series_splits(issues: list[str], n_splits: int = 3, min_train_issues: int = 30) -> list[tuple[list[str], list[str]]]:
    unique = list(dict.fromkeys(issues))
    if len(unique) < min_train_issues + n_splits + 1:
        raise DataContractError("insufficient issues for time-series split")
    val_size = max(10, len(unique) // (n_splits + 1))
    splits: list[tuple[list[str], list[str]]] = []
    for i in range(n_splits):
        train_end = min_train_issues + i * val_size
        val_end = min(train_end + val_size, len(unique))
        train_issues = unique[:train_end]
        val_issues = unique[train_end:val_end]
        if train_issues and val_issues:
            splits.append((train_issues, val_issues))
    if not splits:
        raise DataContractError("failed to create any valid time-series split")
    return splits


def fit_models(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[LGBMRanker, LogisticRegression]:
    ranker = LGBMRanker(
        objective="lambdarank",
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    x_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["label"]
    ranker.fit(x_train, y_train, group=_groups(train_df))

    ranker_score = ranker.predict(x_train)
    lr_x = x_train.copy()
    lr_x["ranker_score"] = ranker_score
    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(lr_x, y_train)
    return ranker, logistic


def score_with_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    ranker: LGBMRanker,
    logistic: LogisticRegression,
    weights: RuntimeWeights,
) -> pd.DataFrame:
    x = df[feature_cols].fillna(0.0)
    ranker_score = ranker.predict(x)
    lr_x = x.copy()
    lr_x["ranker_score"] = ranker_score
    logistic_score = logistic.predict_proba(lr_x)[:, 1]
    table = score_candidates(df, ranker_score, logistic_score, weights)
    return table


def run_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    weights: RuntimeWeights,
    n_splits: int = 3,
    min_train_issues: int = 30,
) -> list[FoldResult]:
    splits = make_time_series_splits(df["issue"].tolist(), n_splits=n_splits, min_train_issues=min_train_issues)
    out: list[FoldResult] = []
    for i, (train_issues, val_issues) in enumerate(splits, 1):
        train_df = df[df["issue"].isin(train_issues)].copy()
        val_df = df[df["issue"].isin(val_issues)].copy()
        ranker, logistic = fit_models(train_df, feature_cols)
        out.append(
            FoldResult(
                fold_id=i,
                train_scored=score_with_models(train_df, feature_cols, ranker, logistic, weights),
                val_scored=score_with_models(val_df, feature_cols, ranker, logistic, weights),
                train_issues=train_issues,
                val_issues=val_issues,
            )
        )
    return out


def compute_metrics(scored: pd.DataFrame, score_col: str = "final_score") -> dict[str, float]:
    per_issue: list[dict[str, float]] = []
    for _, g in scored.groupby("issue"):
        g2 = g.sort_values(score_col, ascending=False)
        top20 = g2.head(20)
        top10 = g2.head(10)
        top5 = g2.head(5)
        top3 = g2.head(3)
        positives = set(g[g["label"] == 1]["candidate_number"].astype(int))

        def hits(df_top: pd.DataFrame) -> int:
            return int(df_top["label"].sum())

        t3_nums = top3["candidate_number"].astype(int).tolist()
        min_dists = [min(abs(n - p) for p in positives) for n in t3_nums] if positives else [79, 79, 79]

        y_true = [g.sort_values("candidate_number")["label"].to_numpy()]
        y_score = [g.sort_values("candidate_number")[score_col].to_numpy()]

        per_issue.append(
            {
                "top20_hit_rate": hits(top20) / 20.0,
                "top10_hit_rate": hits(top10) / 10.0,
                "top5_hit_rate": hits(top5) / 5.0,
                "top3_hit_rate": hits(top3) / 3.0,
                "top3_at_least_one_hit_rate": 1.0 if hits(top3) >= 1 else 0.0,
                "ndcg@10": float(ndcg_score(y_true, y_score, k=10)),
                "exact_hit@3": 1.0 if hits(top3) == 3 else 0.0,
                "exact_hit@10": 1.0 if hits(top10) == 10 else 0.0,
                "adj_hit_pm1@3": float(np.mean([1.0 if d <= 1 else 0.0 for d in min_dists])),
                "strict_adj_only_pm1@3": 1.0 if all(d == 1 for d in min_dists) else 0.0,
                "mean_min_distance_at_3": float(np.mean(min_dists)),
            }
        )
    frame = pd.DataFrame(per_issue)
    if frame.empty:
        raise DataContractError("cannot compute metrics for empty scored frame")
    return {k: float(v) for k, v in frame.mean(numeric_only=True).to_dict().items()}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def baseline_frequency_scores(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    priors = train_df.groupby("candidate_number")["label"].mean().to_dict()
    out = target_df.copy()
    out["final_score"] = out["candidate_number"].map(priors).fillna(0.0)
    return out


def summarize_fold_dispersion(metrics_by_fold: Iterable[dict[str, float]], key: str = "top3_hit_rate") -> float:
    vals = [m.get(key, 0.0) for m in metrics_by_fold]
    return float(np.std(vals)) if vals else 0.0


def metadata_payload(
    feature_cols: list[str],
    train_issues: list[str],
    config: dict[str, Any],
    backtest_summary: dict[str, float],
) -> dict[str, Any]:
    cfg_raw = json.dumps(config, ensure_ascii=False, sort_keys=True)
    return {
        "model_family": "LightGBMRanker+LogisticRegression",
        "training_issues": {
            "count": len(train_issues),
            "first": train_issues[0],
            "last": train_issues[-1],
        },
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "config_snapshot": config,
        "config_hash": hashlib.sha256(cfg_raw.encode("utf-8")).hexdigest(),
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "backtest_summary": backtest_summary,
    }

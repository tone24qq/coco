from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score

from src.io_utils import safe_read_table
from src.runtime_scoring import DynamicWeightConfig, RuntimeWeights, score_candidates
from src.strategy import apply_top3_group_dedup
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
    df = safe_read_table(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataContractError(f"ranking dataset missing required columns: {sorted(missing)}")
    if df.empty:
        raise DataContractError("ranking dataset is empty")

    for issue, grp in df.groupby("issue"):
        nums = grp["candidate_number"].astype(int).tolist()
        if len(nums) != 80 or sorted(nums) != list(range(1, 81)):
            raise DataContractError(f"group ranking contract violated on issue={issue}: candidates must be 1..80 each once")
        if int(grp["label"].sum()) != 20:
            raise DataContractError(f"group ranking contract violated on issue={issue}: label sum must be 20")
    return df


def resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    if not cols:
        raise DataContractError("no feature columns resolved")
    return cols


def _groups(df: pd.DataFrame) -> list[int]:
    _, counts = np.unique(df["issue"].to_numpy(), return_counts=True)
    return counts.astype(int).tolist()


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


def _new_ranker() -> LGBMRanker:
    return LGBMRanker(objective="lambdarank", n_estimators=180, learning_rate=0.05, num_leaves=31, random_state=42)


def _oof_ranker_scores(train_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    issues = list(dict.fromkeys(train_df["issue"].tolist()))
    step = max(10, len(issues) // 5)
    oof = pd.Series(np.nan, index=train_df.index, dtype=float)
    x_all = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_all = train_df["label"].to_numpy(dtype=np.float64)
    issue_arr = train_df["issue"].to_numpy()
    issue_to_rows = {issue: np.flatnonzero(issue_arr == issue) for issue in issues}
    for end in range(step, len(issues), step):
        tr_issues = issues[:end]
        va_issues = issues[end : min(end + step, len(issues))]
        if not va_issues:
            continue
        tr_idx = np.concatenate([issue_to_rows[i] for i in tr_issues]) if tr_issues else np.array([], dtype=np.int64)
        va_idx = np.concatenate([issue_to_rows[i] for i in va_issues]) if va_issues else np.array([], dtype=np.int64)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        ranker = _new_ranker()
        ranker.fit(x_all[tr_idx], y_all[tr_idx], group=[len(issue_to_rows[i]) for i in tr_issues])
        preds = ranker.predict(x_all[va_idx])
        oof.iloc[va_idx] = preds
    return oof


def fit_models(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[LGBMRanker, LogisticRegression]:
    ranker = _new_ranker()
    x_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["label"]

    oof_ranker = _oof_ranker_scores(train_df, feature_cols)
    lr_train = x_train.copy()
    lr_train["ranker_score"] = oof_ranker
    keep = lr_train["ranker_score"].notna()
    if keep.sum() < 200:
        raise DataContractError("insufficient OOF samples for logistic auxiliary training")

    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(lr_train.loc[keep], y_train.loc[keep])

    ranker.fit(x_train, y_train, group=_groups(train_df))
    return ranker, logistic


def score_with_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    ranker: LGBMRanker,
    logistic: LogisticRegression,
    weights: RuntimeWeights,
    dynamic_cfg: DynamicWeightConfig | None = None,
) -> pd.DataFrame:
    x = df[feature_cols].fillna(0.0)
    ranker_score = ranker.predict(x)
    lr_x = x.copy()
    lr_x["ranker_score"] = ranker_score
    logistic_score = logistic.predict_proba(lr_x)[:, 1]
    table = score_candidates(df, ranker_score, logistic_score, weights, dynamic_cfg=dynamic_cfg)
    return table


def run_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    weights: RuntimeWeights,
    dynamic_cfg: DynamicWeightConfig | None = None,
    n_splits: int = 3,
    min_train_issues: int = 30,
) -> list[FoldResult]:
    splits = make_time_series_splits(df["issue"].tolist(), n_splits=n_splits, min_train_issues=min_train_issues)
    issue_arr = df["issue"].to_numpy()
    unique_issue = list(dict.fromkeys(issue_arr.tolist()))
    issue_to_rows = {issue: np.flatnonzero(issue_arr == issue) for issue in unique_issue}
    out: list[FoldResult] = []
    for i, (train_issues, val_issues) in enumerate(splits, 1):
        train_idx = np.concatenate([issue_to_rows[x] for x in train_issues]) if train_issues else np.array([], dtype=np.int64)
        val_idx = np.concatenate([issue_to_rows[x] for x in val_issues]) if val_issues else np.array([], dtype=np.int64)
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        ranker, logistic = fit_models(train_df, feature_cols)
        out.append(
            FoldResult(
                fold_id=i,
                train_scored=score_with_models(train_df, feature_cols, ranker, logistic, weights, dynamic_cfg=dynamic_cfg),
                val_scored=score_with_models(val_df, feature_cols, ranker, logistic, weights, dynamic_cfg=dynamic_cfg),
                train_issues=train_issues,
                val_issues=val_issues,
            )
        )
    return out


def _compute_metrics_legacy(scored: pd.DataFrame, score_col: str = "final_score") -> dict[str, float]:
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

        pre_dedup = t3_nums
        post_dedup = apply_top3_group_dedup(g2.head(10)["candidate_number"].astype(int).tolist())
        per_issue.append(
            {
                "top20_hit_rate": hits(top20) / 20.0,
                "top10_hit_rate": hits(top10) / 10.0,
                "top5_hit_rate": hits(top5) / 5.0,
                "top3_hit_rate": hits(top3) / 3.0,
                "top3_at_least_one_hit_rate": 1.0 if hits(top3) >= 1 else 0.0,
                "top3_at_least_one_hit": 1.0 if hits(top3) >= 1 else 0.0,
                "ndcg@10": float(ndcg_score(y_true, y_score, k=10)),
                "exact_hit@3": 1.0 if hits(top3) == 3 else 0.0,
                "exact_hit@10": 1.0 if hits(top10) == 10 else 0.0,
                "adj_hit_pm1@3": float(np.mean([1.0 if d <= 1 else 0.0 for d in min_dists])),
                "strict_adj_only_pm1@3": 1.0 if all(d == 1 for d in min_dists) else 0.0,
                "mean_min_distance_at_3": float(np.mean(min_dists)),
                "top3_dedup_changed": 1.0 if pre_dedup != post_dedup[:3] else 0.0,
            }
        )
    frame = pd.DataFrame(per_issue)
    if frame.empty:
        raise DataContractError("cannot compute metrics for empty scored frame")
    return {k: float(v) for k, v in frame.mean(numeric_only=True).to_dict().items()}


def compute_metrics(scored: pd.DataFrame, score_col: str = "final_score") -> dict[str, float]:
    if scored.empty:
        raise DataContractError("cannot compute metrics for empty scored frame")
    ordered = scored.sort_values(["issue", score_col], ascending=[True, False])
    by_num = scored.sort_values(["issue", "candidate_number"], ascending=[True, True])
    issues = ordered["issue"].to_numpy()
    labels = ordered["label"].to_numpy(dtype=np.float64)
    cand = ordered["candidate_number"].to_numpy(dtype=np.int64)
    uniq, starts, counts = np.unique(issues, return_index=True, return_counts=True)
    y_true_by_issue = {
        issue: grp["label"].to_numpy(dtype=np.float64)
        for issue, grp in by_num.groupby("issue", sort=False)
    }
    y_score_by_issue = {
        issue: grp[score_col].to_numpy(dtype=np.float64)
        for issue, grp in by_num.groupby("issue", sort=False)
    }

    metrics_rows: list[dict[str, float]] = []
    for issue, st, cnt in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
        ed = st + cnt
        label_slice = labels[st:ed]
        c = cand[st:ed]
        top20 = float(np.sum(label_slice[:20]) / 20.0)
        top10_hits = float(np.sum(label_slice[:10]))
        top5 = float(np.sum(label_slice[:5]) / 5.0)
        top3_hits = float(np.sum(label_slice[:3]))
        top3 = top3_hits / 3.0
        positives = set(c[label_slice == 1.0].tolist())
        t3_nums = c[:3].tolist()
        min_dists = [min(abs(int(n) - int(p)) for p in positives) for n in t3_nums] if positives else [79, 79, 79]
        pre_dedup = t3_nums
        post_dedup = apply_top3_group_dedup(c[:10].astype(int).tolist())
        metrics_rows.append(
            {
                "top20_hit_rate": top20,
                "top10_hit_rate": top10_hits / 10.0,
                "top5_hit_rate": top5,
                "top3_hit_rate": top3,
                "top3_at_least_one_hit_rate": 1.0 if top3_hits >= 1 else 0.0,
                "top3_at_least_one_hit": 1.0 if top3_hits >= 1 else 0.0,
                "ndcg@10": float(ndcg_score([y_true_by_issue[issue]], [y_score_by_issue[issue]], k=10)),
                "exact_hit@3": 1.0 if top3_hits == 3 else 0.0,
                "exact_hit@10": 1.0 if top10_hits == 10 else 0.0,
                "adj_hit_pm1@3": float(np.mean([1.0 if d <= 1 else 0.0 for d in min_dists])),
                "strict_adj_only_pm1@3": 1.0 if all(d == 1 for d in min_dists) else 0.0,
                "mean_min_distance_at_3": float(np.mean(min_dists)),
                "top3_dedup_changed": 1.0 if pre_dedup != post_dedup[:3] else 0.0,
            }
        )
    frame = pd.DataFrame(metrics_rows)
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
    feat_raw = json.dumps(feature_cols, ensure_ascii=False)
    return {
        "model_family": "LightGBMRanker+LogisticRegression",
        "model_version": hashlib.sha256((cfg_raw + feat_raw).encode("utf-8")).hexdigest()[:16],
        "training_issues": {
            "count": len(train_issues),
            "first": train_issues[0],
            "last": train_issues[-1],
        },
        "train_issue_range": [train_issues[0], train_issues[-1]],
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "feature_hash": hashlib.sha256(feat_raw.encode("utf-8")).hexdigest(),
        "config_snapshot": config,
        "config_hash": hashlib.sha256(cfg_raw.encode("utf-8")).hexdigest(),
        "weights": config.get("train", {}).get("runtime_scoring", {}).get("weights", {}),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backtest_summary": backtest_summary,
        "score_semantics": {"final_score": "ranking score", "logistic_score": "auxiliary score"},
    }

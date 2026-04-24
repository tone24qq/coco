from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.split_ranking_dataset import split_df  # noqa: E402
from src.features.weak_signal import build_weak_signal_features  # noqa: E402
from src.safe_io import read_dataset_auto  # noqa: E402


EPS = 1e-12
METRIC_KEYS = ["top1", "top3", "top5", "top10", "mrr", "mean_rank", "normalized_mean_rank_gain"]
REQUIRED_COLUMNS = ["group_id", "label", "is_feasible", "cand_row", "cand_col"]


def _read_auto_compat(path: Path) -> pd.DataFrame:
    try:
        return read_dataset_auto(path)
    except Exception:
        manifest = path / "manifest.json"
        if not manifest.exists():
            raise
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        files = payload.get("files", [])
        if not files:
            raise
        frames: List[pd.DataFrame] = []
        for raw in files:
            fp = Path(str(raw).replace("\\", "/"))
            if not fp.is_absolute():
                fp = ROOT / fp
            frames.append(pd.read_parquet(fp))
        return pd.concat(frames, ignore_index=True)


def _load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_splits(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    if args.train_path and args.valid_path and args.holdout_path:
        return {
            "train": _read_auto_compat(Path(args.train_path)),
            "valid": _read_auto_compat(Path(args.valid_path)),
            "holdout": _read_auto_compat(Path(args.holdout_path)),
        }

    if not args.dataset_path:
        raise ValueError("must provide either dataset-path or train/valid/holdout paths")

    dataset = _read_auto_compat(Path(args.dataset_path))
    split_root = Path(args.split_root) if args.split_root else Path("data/ranking/splits")
    train_p = split_root / "train.parquet"
    valid_p = split_root / "valid.parquet"
    holdout_p = split_root / "holdout.parquet"
    if train_p.exists() and valid_p.exists() and holdout_p.exists():
        return {
            "train": _read_auto_compat(train_p),
            "valid": _read_auto_compat(valid_p),
            "holdout": _read_auto_compat(holdout_p),
        }

    return split_df(
        dataset,
        holdout_ratio=0.2,
        split_mode="by_lineage",
        seed=42,
        include_synth_in_holdout=False,
        valid_real_only=False,
        holdout_real_only=False,
        exclude_synth_from_valid=False,
    )


def check_lineage_or_board_leakage(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> bool:
    for col in ("lineage_id", "board_id"):
        if col in train_df.columns and col in holdout_df.columns:
            tr = set(train_df[col].dropna().astype(str).tolist())
            ho = set(holdout_df[col].dropna().astype(str).tolist())
            if tr & ho:
                return False
    return True


def _fail_fast_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _coerce_binary_label(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)


def _assert_group_integrity(df: pd.DataFrame) -> Tuple[int, int, int]:
    label_sum = _coerce_binary_label(df).groupby(df["group_id"]).sum()
    one = int((label_sum == 1).sum())
    zero = int((label_sum == 0).sum())
    multi = int((label_sum > 1).sum())
    if one != int(label_sum.shape[0]):
        raise ValueError(f"group integrity failed: one={one} zero={zero} multi={multi}")
    return one, zero, multi


def _assert_candidate_counts(df: pd.DataFrame, feasible_counts_all: pd.Series) -> None:
    group_sizes = df.groupby("group_id").size()
    if int(group_sizes.min()) < 2:
        raise ValueError("each group must have at least 2 feasible candidates")
    aligned = feasible_counts_all.reindex(group_sizes.index)
    if aligned.isna().any():
        raise ValueError("candidate_count check failed: missing feasible counts")
    if not (aligned.astype(int) == group_sizes.astype(int)).all():
        raise ValueError("candidate_count mismatch: feasible candidate count mismatch")


def _assert_signal_columns_no_label(df: pd.DataFrame) -> None:
    signal_candidate_cols = [
        c
        for c in df.columns
        if c.startswith("board_state_")
        or c.startswith("candidate_delta_")
        or c in {
            "is_border",
            "is_corner",
            "row_norm",
            "col_norm",
            "dist_to_center",
            "module_consensus_top1",
            "module_consensus_top3",
            "module_consensus_top5",
            "mean_score",
            "std_score",
            "score_spread",
            "disagreement_count",
            "rank_entropy_like",
            "conflict_mass",
        }
    ]
    if "label" in signal_candidate_cols:
        raise AssertionError("label must not be part of weak signal feature columns")


def _metrics_like_train_local_ranker(df: pd.DataFrame, score_col: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    ranked = df[["group_id", "label", score_col]].copy()
    ranked["label"] = _coerce_binary_label(ranked)

    top_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    rows: List[Dict[str, Any]] = []
    ranks: List[int] = []
    for gid, g in ranked.groupby("group_id", sort=False):
        gg = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        pos = gg.index[gg["label"] == 1].tolist()
        if not pos:
            continue
        rank = int(pos[0] + 1)
        ranks.append(rank)
        for k in top_hits:
            top_hits[k] += int(rank <= k)
        rows.append({"group_id": gid, "rank": rank, "candidate_count": int(len(gg)), "rr": 1.0 / rank})

    per_group = pd.DataFrame(rows)
    total = max(len(ranks), 1)
    cand_mean = float(df.groupby("group_id").size().mean()) if len(df) else 0.0
    mean_rank = float(np.mean(ranks)) if ranks else 0.0
    mrr = float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0
    norm_gain = 1.0
    if cand_mean > 1.0:
        norm_gain = float(1.0 - (mean_rank - 1.0) / (cand_mean - 1.0))

    metrics = {
        "group_count": int(total),
        "top1": top_hits[1] / total,
        "top3": top_hits[3] / total,
        "top5": top_hits[5] / total,
        "top10": top_hits[10] / total,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "normalized_mean_rank_gain": norm_gain,
    }
    return metrics, per_group


def _predict_with_artifact(df: pd.DataFrame, artifact_path: Path) -> Tuple[np.ndarray | None, str]:
    if not artifact_path.exists():
        return None, "missing_model_artifact"
    artifact = joblib.load(artifact_path)
    model = artifact.get("model")
    feature_columns = list(artifact.get("feature_columns", []))
    if model is None or not feature_columns:
        return None, "missing_model_artifact"
    x = np.asarray(df.reindex(columns=feature_columns, fill_value=0.0).fillna(0.0), dtype=np.float32)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float), "artifact_predict"
    return np.asarray(model.predict(x), dtype=float), "artifact_predict"


def _temp_feature_columns(df: pd.DataFrame) -> List[str]:
    forbidden = {"label", "group_id", "target_number", "board_id", "lineage_id", "cand_row", "cand_col"}
    cols = [
        c
        for c in df.columns
        if (c.startswith("board_state_") or c.startswith("candidate_delta_"))
        and c not in forbidden
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(cols)


def _fit_temp_model_predict(train_df: pd.DataFrame, score_df: pd.DataFrame) -> Tuple[np.ndarray | None, str]:
    cols = _temp_feature_columns(train_df)
    if not cols:
        return None, "no_valid_feature_columns"
    x_train = np.asarray(train_df[cols].fillna(0.0), dtype=np.float32)
    y_train = _coerce_binary_label(train_df).to_numpy()
    model = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.06, max_iter=200, random_state=42)
    model.fit(x_train, y_train)
    x_score = np.asarray(score_df.reindex(columns=cols, fill_value=0.0).fillna(0.0), dtype=np.float32)
    return np.asarray(model.predict_proba(x_score)[:, 1], dtype=float), "temp_model_predict"


def _assign_baseline_scores(
    splits: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], str, List[str]]:
    reasons: List[str] = []
    order = ["train", "valid", "holdout"]
    frames = [splits[k].copy() for k in order]
    all_df = pd.concat(frames, ignore_index=True)

    baseline_series = None
    source = ""

    if "baseline_score" in all_df.columns:
        candidate = pd.to_numeric(all_df["baseline_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if float(candidate.std()) > 0.0:
            baseline_series = candidate
            source = "dataset_baseline_score"
        else:
            reasons.append("baseline_score_constant")

    if baseline_series is None:
        pred, state = _predict_with_artifact(all_df, Path("artifacts/global/main_ranker.pkl"))
        if pred is not None:
            baseline_series = pd.Series(pred, index=all_df.index)
            source = state
        else:
            reasons.append(state)

    if baseline_series is None:
        train_df = splits["train"]
        pred, state = _fit_temp_model_predict(train_df, all_df)
        if pred is None:
            reasons.append(state)
            return splits, "", reasons
        baseline_series = pd.Series(pred, index=all_df.index)
        source = state

    baseline_series = pd.to_numeric(baseline_series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(baseline_series.std()) <= 0.0:
        reasons.append("baseline_score_constant")

    all_df["baseline_score"] = baseline_series
    out: Dict[str, pd.DataFrame] = {}
    offset = 0
    for name in order:
        n = len(splits[name])
        out[name] = all_df.iloc[offset : offset + n].copy().reset_index(drop=True)
        offset += n
    return out, source, reasons


def _build_debug_payload(df_all: pd.DataFrame) -> Dict[str, Any]:
    labels = _coerce_binary_label(df_all)
    label_sum = int(labels.sum())
    if "group_id" in df_all.columns:
        group_sizes = df_all.groupby("group_id").size()
        label_group = labels.groupby(df_all["group_id"]).sum()
        group_count = int(group_sizes.shape[0])
        groups_one = int((label_group == 1).sum())
        groups_zero = int((label_group == 0).sum())
        groups_multi = int((label_group > 1).sum())
        cand_min = int(group_sizes.min()) if len(group_sizes) else 0
        cand_mean = float(group_sizes.mean()) if len(group_sizes) else 0.0
        cand_max = int(group_sizes.max()) if len(group_sizes) else 0
    else:
        group_count = 0
        groups_one = 0
        groups_zero = 0
        groups_multi = 0
        cand_min = 0
        cand_mean = 0.0
        cand_max = 0

    feature_col_count = len([c for c in df_all.columns if c.startswith("board_state_") or c.startswith("candidate_delta_")])
    baseline_exists = "baseline_score" in df_all.columns
    baseline_std = float(pd.to_numeric(df_all.get("baseline_score", pd.Series([0])), errors="coerce").std()) if baseline_exists else 0.0
    weak_std = float(pd.to_numeric(df_all.get("weak_signal_score", pd.Series([0])), errors="coerce").std()) if "weak_signal_score" in df_all.columns else 0.0

    return {
        "row_count": int(len(df_all)),
        "group_count": group_count,
        "columns": list(df_all.columns),
        "label_sum": label_sum,
        "label_value_counts": {str(k): int(v) for k, v in labels.value_counts(dropna=False).to_dict().items()},
        "is_feasible_value_counts": {
            str(k): int(v)
            for k, v in pd.to_numeric(df_all["is_feasible"], errors="coerce").fillna(-1).value_counts(dropna=False).to_dict().items()
        }
        if "is_feasible" in df_all.columns
        else {},
        "groups_with_one_positive_label": groups_one,
        "groups_without_positive_label": groups_zero,
        "groups_with_multiple_positive_label": groups_multi,
        "candidate_count_min": cand_min,
        "candidate_count_mean": cand_mean,
        "candidate_count_max": cand_max,
        "feature_column_count": feature_col_count,
        "baseline_score_exists": baseline_exists,
        "baseline_score_std": baseline_std,
        "weak_signal_score_std": weak_std,
    }


def _size_metrics(per_group: pd.DataFrame) -> pd.DataFrame:
    if per_group.empty:
        return pd.DataFrame(columns=["weight", "size_class", "group_count", "top1", "top3", "top5", "top10", "mrr", "mean_rank"])
    return (
        per_group.groupby(["weight", "size_class"], dropna=False)
        .agg(
            group_count=("group_id", "count"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            top10=("top10", "mean"),
            mrr=("rr", "mean"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
    )


def run_ablation(splits: Dict[str, pd.DataFrame], config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_reasons: List[str] = []

    combined_raw = pd.concat([splits["train"], splits["valid"], splits["holdout"]], ignore_index=True)
    _fail_fast_required_columns(combined_raw)

    if int(_coerce_binary_label(combined_raw).sum()) <= 0:
        failure_reasons.append("no_positive_labels")

    feasible_counts = (
        combined_raw[pd.to_numeric(combined_raw["is_feasible"], errors="coerce").fillna(0).astype(int) == 1]
        .groupby("group_id")
        .size()
    )

    splits = {
        k: v[pd.to_numeric(v["is_feasible"], errors="coerce").fillna(0).astype(int) == 1].copy().reset_index(drop=True)
        for k, v in splits.items()
    }
    if sum(len(v) for v in splits.values()) == 0:
        failure_reasons.append("empty_after_filter")

    combined = pd.concat([splits["train"], splits["valid"], splits["holdout"]], ignore_index=True)
    if not combined.empty:
        label_group = _coerce_binary_label(combined).groupby(combined["group_id"]).sum()
        valid_groups = set(label_group[label_group == 1].index.tolist())
        if len(valid_groups) != int(label_group.shape[0]):
            failure_reasons.append("no_positive_labels")
        splits = {k: v[v["group_id"].isin(valid_groups)].copy().reset_index(drop=True) for k, v in splits.items()}
        combined = pd.concat([splits["train"], splits["valid"], splits["holdout"]], ignore_index=True)

    if combined.empty:
        failure_reasons.append("empty_after_filter")
        decision = {
            "accepted": False,
            "best_weight": 0.0,
            "baseline": {k: 0.0 for k in METRIC_KEYS},
            "best": {k: 0.0 for k in METRIC_KEYS},
            "delta": {k: 0.0 for k in METRIC_KEYS},
            "guardrails": {},
            "weak_signal_all_zero": False,
            "baseline_source": "",
            "reason": ";".join(sorted(set(failure_reasons))),
        }
        debug_payload = _build_debug_payload(combined_raw)
        (output_dir / "debug_input.json").write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame().to_csv(output_dir / "summary.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "per_group.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "per_size.csv", index=False)
        (output_dir / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        return decision

    _assert_group_integrity(combined)
    _assert_candidate_counts(combined, feasible_counts)

    splits, baseline_source, baseline_failures = _assign_baseline_scores(splits)
    failure_reasons.extend(baseline_failures)
    combined = pd.concat([splits["train"], splits["valid"], splits["holdout"]], ignore_index=True)
    baseline_std = float(pd.to_numeric(combined["baseline_score"], errors="coerce").std())
    if baseline_std <= 0.0:
        failure_reasons.append("baseline_score_constant")

    _assert_signal_columns_no_label(combined)

    weights = [float(w) for w in config.get("weights", [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20])]
    summary_rows: List[Dict[str, Any]] = []
    per_group_rows: List[pd.DataFrame] = []

    baseline_metrics: Dict[str, float] = {}
    weak_signal_all_zero = False
    baseline_ranks = pd.DataFrame(columns=["group_id", "baseline_rank"])

    for weight in weights:
        split_frames: List[pd.DataFrame] = []
        for split_name in ("train", "valid", "holdout"):
            frame = build_weak_signal_features(splits[split_name], config)
            frame["split"] = split_name
            base = pd.to_numeric(frame["baseline_score"], errors="coerce").fillna(0.0)
            weak = pd.to_numeric(frame["weak_signal_score"], errors="coerce").fillna(0.0)
            frame["ablation_score"] = base + float(weight) * weak
            split_frames.append(frame)

        merged = pd.concat(split_frames, ignore_index=True)
        weak_std = float(pd.to_numeric(merged["weak_signal_score"], errors="coerce").std())
        if weak_std <= 0.0:
            weak_signal_all_zero = True

        if abs(weight) <= EPS and not np.allclose(
            pd.to_numeric(merged["ablation_score"], errors="coerce").to_numpy(),
            pd.to_numeric(merged["baseline_score"], errors="coerce").to_numpy(),
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("weight=0.00 ablation_score != baseline_score")

        metrics, per_group = _metrics_like_train_local_ranker(merged, "ablation_score")
        if abs(weight) <= EPS:
            baseline_metrics = dict(metrics)

        per_group = per_group.merge(
            merged[["group_id", "split", "size_class", "weak_signal_score"]].drop_duplicates("group_id"),
            on="group_id",
            how="left",
        )
        per_group["weight"] = float(weight)
        per_group["top1"] = (per_group["rank"] <= 1).astype(int)
        per_group["top3"] = (per_group["rank"] <= 3).astype(int)
        per_group["top5"] = (per_group["rank"] <= 5).astype(int)
        per_group["top10"] = (per_group["rank"] <= 10).astype(int)

        if abs(weight) <= EPS:
            baseline_ranks = per_group[["group_id", "rank"]].rename(columns={"rank": "baseline_rank"})
        pg = per_group.merge(baseline_ranks, on="group_id", how="left")
        pg["rank_gain_vs_baseline"] = pg["baseline_rank"] - pg["rank"]
        win_rate = float((pg["rank"] < pg["baseline_rank"]).mean())

        split_metrics_rows = []
        for sp, sub in pg.groupby("split"):
            split_metrics_rows.append(
                {
                    "split": str(sp),
                    "group_count": int(len(sub)),
                    "top1": float((sub["rank"] <= 1).mean()),
                    "top3": float((sub["rank"] <= 3).mean()),
                    "top5": float((sub["rank"] <= 5).mean()),
                    "top10": float((sub["rank"] <= 10).mean()),
                    "mrr": float((1.0 / sub["rank"]).mean()),
                    "mean_rank": float(sub["rank"].mean()),
                }
            )

        summary_rows.append(
            {
                "weight": float(weight),
                **metrics,
                "fold_group_win_rate_vs_baseline": win_rate,
                "per_split_metrics": json.dumps(split_metrics_rows, ensure_ascii=False),
            }
        )
        per_group_rows.append(pg)

    summary = pd.DataFrame(summary_rows).sort_values("weight").reset_index(drop=True)
    per_group_all = pd.concat(per_group_rows, ignore_index=True)
    per_size = _size_metrics(per_group_all)

    weight0 = summary[np.isclose(summary["weight"], 0.0)]
    if weight0.empty:
        raise ValueError("missing weight=0.00 row")
    weight0_metrics = weight0.iloc[0].to_dict()
    for key in METRIC_KEYS:
        if abs(float(weight0_metrics[key]) - float(baseline_metrics[key])) > 1e-12:
            raise ValueError(f"weight=0 metrics mismatch for {key}")

    summary.to_csv(output_dir / "summary.csv", index=False)
    per_group_all.to_csv(output_dir / "per_group.csv", index=False)
    per_size.to_csv(output_dir / "per_size.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_idx = summary["mrr"].idxmax()
    best = summary.loc[best_idx].to_dict()
    baseline = weight0.iloc[0].to_dict()
    delta = {k: float(best[k] - baseline[k]) for k in METRIC_KEYS}

    base_size = per_group_all[np.isclose(per_group_all["weight"], 0.0)].groupby("size_class")["rank"].mean()
    best_size = per_group_all[np.isclose(per_group_all["weight"], float(best["weight"]))].groupby("size_class")["rank"].mean()
    improved_sizes = [s for s in sorted(set(base_size.index) & set(best_size.index)) if float(best_size[s]) < float(base_size[s])]
    not_single_size_only = len(improved_sizes) != 1

    guardrails = {
        "top1_not_degraded": bool(delta["top1"] >= -0.005),
        "top3_not_degraded": bool(delta["top3"] >= 0.0),
        "mrr_improved": bool(delta["mrr"] > 0.0),
        "mean_rank_improved": bool(delta["mean_rank"] <= 0.0),
        "not_single_size_only": bool(not_single_size_only),
        "no_nan_inf": bool(np.isfinite(summary.select_dtypes(include=[np.number]).to_numpy()).all()),
        "group_integrity_passed": True,
        "lineage_or_board_leakage_guard_passed": bool(check_lineage_or_board_leakage(splits["train"], splits["holdout"])),
    }

    accepted = bool(
        (delta["top3"] > 0.0 or delta["mrr"] > 0.0)
        and guardrails["top1_not_degraded"]
        and guardrails["mean_rank_improved"]
        and guardrails["not_single_size_only"]
        and guardrails["no_nan_inf"]
        and guardrails["group_integrity_passed"]
        and guardrails["lineage_or_board_leakage_guard_passed"]
    )

    if weak_signal_all_zero:
        failure_reasons.append("weak_signal_score_constant")

    decision = {
        "accepted": accepted,
        "best_weight": float(best["weight"]),
        "baseline": {k: float(baseline[k]) for k in METRIC_KEYS},
        "best": {k: float(best[k]) for k in METRIC_KEYS},
        "delta": delta,
        "guardrails": guardrails,
        "weak_signal_all_zero": bool(weak_signal_all_zero),
        "baseline_source": baseline_source,
        "reason": ";".join(sorted(set(failure_reasons))) if failure_reasons else ("accepted" if accepted else "no_top3_or_mrr_improvement"),
    }

    debug_payload = _build_debug_payload(pd.concat([splits["train"], splits["valid"], splits["holdout"]], ignore_index=True))
    debug_payload["baseline_score_std"] = float(pd.to_numeric(combined["baseline_score"], errors="coerce").std())
    debug_payload["weak_signal_score_std"] = float(pd.to_numeric(per_group_all["weak_signal_score"], errors="coerce").std())
    (output_dir / "debug_input.json").write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    (output_dir / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = [
        "# weak_signal ablation report",
        "",
        f"- accepted: {accepted}",
        f"- best_weight: {best['weight']}",
        f"- reason: {decision['reason']}",
    ]
    if not accepted:
        readme.extend(["", "weak_signal_score 暫不接入正式 inference pipeline。"])
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    return decision


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--split-root", default="data/ranking/splits")
    parser.add_argument("--train-path", default="")
    parser.add_argument("--valid-path", default="")
    parser.add_argument("--holdout-path", default="")
    parser.add_argument("--config", default="configs/weak_signal.yaml")
    parser.add_argument("--output-dir", default="reports/weak_signal_ablation")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = _load_config(Path(args.config))
    splits = _load_splits(args)
    decision = run_ablation(splits, config, Path(args.output_dir))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

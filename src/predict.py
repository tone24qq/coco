from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.artifacts import ModelArtifacts, load_artifacts
from src.build_features import build_candidate_rows
from src.fetch_winwin import fetch_latest
from src.runtime_scoring import RuntimeWeights, score_candidates
from src.strategy import apply_top3_group_dedup
from src.utils import DataContractError, DrawRecord, ensure_numbers, parse_date, read_processed


def _next_issue(issue: str) -> str:
    if issue.isdigit():
        return str(int(issue) + 1)
    return f"{issue}_next"


def _records_from_payload(recent_draws: list[dict[str, Any]]) -> list[DrawRecord]:
    records = []
    for row in recent_draws:
        records.append(
            DrawRecord(
                issue=str(row["issue"]),
                draw_date=parse_date(str(row["draw_date"])),
                numbers=ensure_numbers(row["numbers"]),
                day_issue_index=int(row["day_issue_index"]),
            )
        )
    if not records:
        raise DataContractError("recent_draws is empty")
    return records


def _load_recent_draws(config: dict[str, Any], recent_draws: list[dict[str, Any]] | None) -> tuple[list[DrawRecord], str]:
    if recent_draws:
        return _records_from_payload(recent_draws), "manual"

    if config.get("auto_fetch", {}).get("enabled", True):
        sources = config.get("auto_fetch", {}).get("sources")
        fetched = fetch_latest(sources=sources)
        lookback = int(config.get("history", {}).get("lookback_n", 150))
        return fetched.records[-lookback:], "winwin_auto_fetch"

    processed = Path(config.get("history", {}).get("processed_path", "data/processed/history_processed.csv"))
    records = read_processed(processed)
    lookback = int(config.get("history", {}).get("lookback_n", 150))
    return records[-lookback:], "processed_history"


def _validate_feature_contract(feature_df: pd.DataFrame, artifacts: ModelArtifacts) -> None:
    missing = [c for c in artifacts.feature_columns if c not in feature_df.columns]
    if missing:
        raise DataContractError(f"feature column mismatch, missing: {missing[:10]}")


def run_prediction(
    artifacts: ModelArtifacts,
    config: dict[str, Any],
    recent_draws: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    history, source = _load_recent_draws(config, recent_draws)
    if len(history) < int(config.get("history", {}).get("lookback_n", 100)):
        raise DataContractError("insufficient recent draws for prediction")

    target_issue = _next_issue(history[-1].issue)
    rows = build_candidate_rows(
        history=history,
        issue=target_issue,
        draw_date=history[-1].draw_date.isoformat(),
        label_numbers=None,
        retrieval_window=min(int(config.get("history", {}).get("lookback_n", 100)), len(history)),
        top_k=int(config.get("retrieval", {}).get("top_k", 20)),
    )
    feat_df = pd.DataFrame(rows)
    if len(feat_df) != 80:
        raise DataContractError("prediction contract violated: expected 80 candidates")

    _validate_feature_contract(feat_df, artifacts)

    x = feat_df[artifacts.feature_columns].fillna(0.0)
    ranker_score = artifacts.ranker.predict(x)
    lr_x = x.copy()
    lr_x["ranker_score"] = ranker_score
    logistic_score = artifacts.logistic.predict_proba(lr_x)[:, 1]

    weights = RuntimeWeights.from_mapping(config.get("runtime_scoring", {}).get("weights", {}))
    scored = score_candidates(feat_df, ranker_score, logistic_score, weights)

    top20 = scored.head(20)["candidate_number"].astype(int).tolist()
    top10 = top20[:10]
    top3_before = top10[:3]
    top3_after = apply_top3_group_dedup(top10)

    table = scored[
        [
            "candidate_number",
            "rank_final",
            "final_score",
            "ranker_score",
            "logistic_score",
            "retrieval_score",
            "history_prior_score",
            "analysis_rerank_score",
            "local_peak_score",
        ]
    ].rename(columns={"candidate_number": "number"})

    return {
        "issue": target_issue,
        "source": source,
        "top20_numbers": top20,
        "top10_numbers": top10,
        "top3_numbers": top3_after,
        "top3_before_group_dedup": top3_before,
        "top3_after_group_dedup": top3_after,
        "ranking_score_table": table.to_dict(orient="records"),
        "metadata": {
            "model_family": artifacts.metadata.get("model_family", "unknown"),
            "model_version": artifacts.metadata.get("created_at", "unknown"),
            "feature_count": len(artifacts.feature_columns),
            "score_type": "ranking_score",
            "auxiliary_score": "logistic_score",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/predict.yaml")
    parser.add_argument("--output", default="reports/latest_prediction.json")
    parser.add_argument("--recent-json", default="")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    artifacts = load_artifacts(Path(config.get("models", {}).get("dir", "models")))

    recent_draws = None
    if args.recent_json:
        recent_draws = json.loads(Path(args.recent_json).read_text(encoding="utf-8"))

    result = run_prediction(artifacts, config, recent_draws)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except DataContractError as exc:
        raise SystemExit(f"[fail-fast] {exc}")

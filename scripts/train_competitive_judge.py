from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression

from src.inference_config import load_aggregator_config, load_module_weights
from src.inference_service import _run_inference_detailed


def _load_real_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        required = {"board", "target_number", "true_cell"}
        if not required.issubset(set(obj.keys())):
            raise ValueError(f"real case missing keys: {required - set(obj.keys())}")
        true_cell = tuple(obj["true_cell"])
        rows.append({"board": obj["board"], "target": int(obj["target_number"]), "true": true_cell})
    return rows


def _extract_features(out: Dict[str, Any], true_cell: Tuple[int, int]) -> Tuple[List[List[float]], List[int], List[str]]:
    names = [
        "mean_score",
        "std_score",
        "score_spread",
        "top1_vote_count",
        "top3_vote_count",
        "top5_vote_count",
        "borda_score",
        "rrf_score",
        "disagreement_count",
        "rank_entropy_like",
        "support_margin_to_next",
        "conflict_mass",
    ]
    X: List[List[float]] = []
    y: List[int] = []
    for cand in out["candidate_cells"]:
        X.append([float(cand.get(name, 0.0)) for name in names])
        y.append(1 if (cand["row"], cand["col"]) == true_cell else 0)
    return X, y, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", default="data/competitive_cases.jsonl")
    parser.add_argument("--min-real-cases", type=int, default=8)
    args = parser.parse_args()

    real_cases = _load_real_cases(Path(args.real_data))
    if len(real_cases) < args.min_real_cases:
        raise SystemExit(f"insufficient real data: got {len(real_cases)}, need >= {args.min_real_cases}")

    weights = load_module_weights()
    cfg = load_aggregator_config()
    cfg["type"] = "competitive_ensemble"
    cfg["fusion_mode"] = "weighted_rank_fusion"

    fold_results: List[Dict[str, Any]] = []
    all_X: List[List[float]] = []
    all_y: List[int] = []
    feature_names: List[str] = []

    for i in range(2, len(real_cases)):
        train_cases = real_cases[:i]
        valid_case = real_cases[i]
        X_train: List[List[float]] = []
        y_train: List[int] = []
        for case in train_cases:
            out = _run_inference_detailed(
                case["board"],
                case["target"],
                source="train_competitive_judge",
                module_weights=weights,
                apply_reranker_stage=False,
                include_module_details=False,
                aggregator_config=cfg,
            )
            X, y, names = _extract_features(out, case["true"])
            if not feature_names:
                feature_names = names
            X_train.extend(X)
            y_train.extend(y)
            all_X.extend(X)
            all_y.extend(y)
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        out_valid = _run_inference_detailed(
            valid_case["board"],
            valid_case["target"],
            source="train_competitive_judge_valid",
            module_weights=weights,
            apply_reranker_stage=False,
            include_module_details=False,
            aggregator_config=cfg,
        )
        rank = 999
        for idx, cand in enumerate(out_valid["candidate_cells"], start=1):
            fv = [float(cand.get(n, 0.0)) for n in feature_names]
            score = float(model.intercept_[0] + sum(v * w for v, w in zip(fv, model.coef_[0])))
            cand["_judge_score"] = score
        ranked = sorted(out_valid["candidate_cells"], key=lambda c: c["_judge_score"], reverse=True)
        for idx, cand in enumerate(ranked, start=1):
            if (cand["row"], cand["col"]) == valid_case["true"]:
                rank = idx
                break
        fold_results.append({"fold_index": i - 1, "train_cases": i, "valid_rank": rank})

    final_model = LogisticRegression(max_iter=500)
    final_model.fit(all_X, all_y)

    created_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "model_type": "logistic_ranker",
        "schema_version": "competitive_features_v1",
        "feature_names": feature_names,
        "coef": [float(x) for x in final_model.coef_[0]],
        "intercept": float(final_model.intercept_[0]),
        "trained_from_real_data": True,
        "walk_forward_only": True,
        "fold_count": len(fold_results),
        "train_case_count": len(real_cases),
        "valid_case_count": max(len(real_cases) - 2, 0),
        "created_at": created_at,
        "producer_script": "scripts/train_competitive_judge.py",
        "fold_metadata": fold_results,
    }
    feature_schema = {
        "schema_version": "competitive_features_v1",
        "feature_names": feature_names,
        "created_at": created_at,
        "producer_script": "scripts/train_competitive_judge.py",
    }
    report = {
        "walk_forward_only": True,
        "trained_from_real_data": True,
        "fold_results": fold_results,
        "top1_rate": sum(1 for x in fold_results if x["valid_rank"] == 1) / max(len(fold_results), 1),
        "mrr": sum(1.0 / x["valid_rank"] for x in fold_results) / max(len(fold_results), 1),
    }

    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("artifacts/competitive_judge_artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    Path("artifacts/competitive_judge_feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2), encoding="utf-8"
    )
    Path("reports/competitive_judge_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("artifacts/competitive_judge_artifact.json")


if __name__ == "__main__":
    main()

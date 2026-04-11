from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_config import load_aggregator_config, load_module_weights  # noqa: E402
from src.inference_service import _run_inference_detailed  # noqa: E402


def _make_case(seed: int, rows: int, cols: int, mask_mod: int = 2) -> Dict[str, Any]:
    board = []
    n = 1
    for r in range(rows):
        row = []
        for c in range(cols):
            value = n
            n += 1
            row.append(-1 if (r + c + seed) % mask_mod == 0 else value)
        board.append(row)
    opened = {v for row in board for v in row if v != -1}
    target = min(x for x in range(1, rows * cols + 1) if x not in opened)
    true = ((target - 1) // cols + 1, (target - 1) % cols + 1)
    return {"board": board, "target": target, "true": true, "shape": f"{rows}x{cols}"}


def _score_case(
    case: Dict[str, Any],
    weights: Dict[str, float],
    aggregator_cfg: Dict[str, Any],
) -> Tuple[int, float, Dict[str, Any]]:
    t0 = time.perf_counter()
    out = _run_inference_detailed(
        case["board"],
        case["target"],
        source="competitive_backtest",
        module_weights=weights,
        apply_reranker_stage=False,
        include_module_details=False,
        aggregator_config=aggregator_cfg,
    )
    latency = (time.perf_counter() - t0) * 1000.0
    rank = 999
    for i, c in enumerate(out["candidate_cells"], start=1):
        if (c["row"], c["col"]) == case["true"]:
            rank = i
            break
    return rank, latency, out


def _metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    n = max(len(rows), 1)
    return {
        "top1_hit_rate": round(sum(1 for r in rows if r["rank"] == 1) / n, 6),
        "top3_hit_rate": round(sum(1 for r in rows if r["rank"] <= 3) / n, 6),
        "top5_hit_rate": round(sum(1 for r in rows if r["rank"] <= 5) / n, 6),
        "mrr": round(sum(1.0 / r["rank"] for r in rows) / n, 6),
        "mean_true_rank": round(sum(r["rank"] for r in rows) / n, 6),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 3),
    }


def _train_meta_artifact(
    cases: List[Dict[str, Any]],
    weights: Dict[str, float],
    base_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    X: List[List[float]] = []
    y: List[int] = []
    feature_names: List[str] = []
    for case in cases:
        cfg = dict(base_cfg)
        cfg["fusion_mode"] = "weighted_rank_fusion"
        _, _, out = _score_case(case, weights, cfg)
        for cand in out["candidate_cells"]:
            row = {
                "mean_score": float(cand.get("mean_score", 0.0)),
                "std_score": float(cand.get("std_score", 0.0)),
                "top1_vote_count": float(cand.get("top1_vote_count", 0.0)),
                "top3_vote_count": float(cand.get("top3_vote_count", 0.0)),
                "borda_score": float(cand.get("borda_score", 0.0)),
                "rrf_score": float(cand.get("rrf_score", 0.0)),
                "disagreement_count": float(cand.get("disagreement_count", 0.0)),
                "conflict_mass": float(cand.get("conflict_mass", 0.0)),
            }
            if not feature_names:
                feature_names = list(row.keys())
            X.append([row[n] for n in feature_names])
            y.append(1 if (cand["row"], cand["col"]) == case["true"] else 0)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    artifact = {
        "model_type": "logistic_ranker",
        "schema_version": "competitive_meta_v1",
        "feature_names": feature_names,
        "coef": [float(x) for x in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "training_split": "walk_forward_time_series",
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/competitive_judge_artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    weights = load_module_weights()
    base_cfg = load_aggregator_config()
    cases = [_make_case(i, 8, 10, 2) for i in range(6)] + [_make_case(100 + i, 10, 16, 3) for i in range(6)]

    all_rows: Dict[str, List[Dict[str, float]]] = {
        "current_hierarchical_fusion": [],
        "equal_weight_average": [],
        "weighted_rank_fusion": [],
        "vote_fusion": [],
        "learned_meta_judge": [],
    }
    feature_importance: Dict[str, float] = {}
    by_shape: Dict[str, Dict[str, List[int]]] = {}

    for idx in range(2, len(cases)):
        train_cases = cases[:idx]
        test_case = cases[idx]
        artifact = _train_meta_artifact(train_cases, weights, dict(base_cfg))
        feature_importance = {n: abs(float(w)) for n, w in zip(artifact["feature_names"], artifact["coef"])}

        for name, mode, agg_type, w in [
            ("current_hierarchical_fusion", "weighted_only", "gate_then_weighted_sum", weights),
            (
                "equal_weight_average",
                "weighted_rank_fusion",
                "competitive_ensemble",
                {k: 1.0 / len(weights) for k in weights},
            ),
            ("weighted_rank_fusion", "weighted_rank_fusion", "competitive_ensemble", weights),
            ("vote_fusion", "vote_based_fusion", "competitive_ensemble", weights),
            ("learned_meta_judge", "learned_meta_ranker", "competitive_ensemble", weights),
        ]:
            cfg = dict(base_cfg)
            cfg["type"] = agg_type
            cfg["fusion_mode"] = mode
            rank, latency, _ = _score_case(test_case, w, cfg)
            all_rows[name].append({"rank": rank, "latency_ms": latency})
            by_shape.setdefault(test_case["shape"], {}).setdefault(name, []).append(rank)

    summary = {k: _metrics(v) for k, v in all_rows.items()}
    preference = {
        "weighted_rank_fusion": 0,
        "learned_meta_judge": 1,
        "vote_fusion": 2,
        "equal_weight_average": 3,
        "current_hierarchical_fusion": 4,
    }
    recommended = min(
        summary.items(),
        key=lambda kv: (
            kv[1]["mean_true_rank"],
            -kv[1]["top1_hit_rate"],
            preference.get(kv[0], 99),
        ),
    )[0]
    report = {
        "walk_forward_split_only": True,
        "comparison": summary,
        "judge_selected_feature_importance": feature_importance,
        "shape_mode_differences": by_shape,
        "recommended_default": recommended,
    }
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports/competitive_fusion_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

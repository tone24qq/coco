from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hit_benchmark import _build_cases_from_parsed_boards, _default_cases
from src.inference_service import _run_inference_detailed
from src.ranking_features import FEATURE_SCHEMA_VERSION, build_candidate_feature_rows, feature_columns_from_rows
from src.reranker import ARTIFACTS_DIR, FEATURE_COLUMNS_PATH, MODEL_PATH, WEIGHTS_PATH


MIN_CASES_FOR_LEARNED = 30


@dataclass
class EvalResult:
    top5: float
    top3: float
    mrr: float


def _eval_rows(rows: List[Dict[str, Any]]) -> EvalResult:
    case_to_rank = {}
    for r in rows:
        if r["label"] == 1:
            case_to_rank[r["case_id"]] = int(r["pred_rank"])
    total = max(len(case_to_rank), 1)
    top5 = sum(1 for v in case_to_rank.values() if v <= 5) / total
    top3 = sum(1 for v in case_to_rank.values() if v <= 3) / total
    mrr = sum(1.0 / v for v in case_to_rank.values()) / total
    return EvalResult(top5=top5, top3=top3, mrr=mrr)


def _score_with_weights(rows: List[Dict[str, Any]], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        s = 0.0
        for k, w in weights.items():
            s += float(r.get(k, 0.0)) * w
        rr = dict(r)
        rr["pred_score"] = s
        out.append(rr)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in out:
        grouped.setdefault(r["case_id"], []).append(r)

    ranked = []
    for case_rows in grouped.values():
        case_rows.sort(key=lambda x: x["pred_score"], reverse=True)
        for idx, rr in enumerate(case_rows, start=1):
            rr["pred_rank"] = idx
            ranked.append(rr)
    return ranked


def _weight_search(train_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    grid = [-1.0, -0.5, 0.0, 0.5, 1.0]
    best = None
    best_metrics = None
    for w_base in grid:
        for w_cons in grid:
            for w_dist in grid:
                weights = {
                    "baseline_score": w_base,
                    "module_consensus_top3": w_cons,
                    "dist_to_center": w_dist,
                }
                scored = _score_with_weights(train_rows, weights)
                metrics = _eval_rows(scored)
                rank_key = (metrics.top5, metrics.top3, metrics.mrr)
                if best is None or rank_key > best_metrics:
                    best = weights
                    best_metrics = rank_key
    return best or {"baseline_score": 1.0, "module_consensus_top3": 0.0, "dist_to_center": 0.0}


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = _build_cases_from_parsed_boards(Path("reports/parsed_boards.json"), max_cases=60, seed=2026)
    if not cases:
        cases = _default_cases(seed=2026)

    all_rows: List[Dict[str, Any]] = []
    for case in cases:
        result = _run_inference_detailed(case.masked_board, case.target_number, source="train_reranker", apply_reranker_stage=False)
        rows = build_candidate_feature_rows(
            case_id=case.sample_id,
            board_shape=(len(case.masked_board), len(case.masked_board[0])),
            candidates=result["candidate_cells"],
            true_cell_1_based=(case.true_cell_0_based[0] + 1, case.true_cell_0_based[1] + 1),
        )
        all_rows.extend(rows)

    feature_cols = feature_columns_from_rows(all_rows)

    if len(cases) < MIN_CASES_FOR_LEARNED:
        payload = {
            "enabled": False,
            "version": "weight_search_v1",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_columns": feature_cols,
            "fallback_reason": f"insufficient_cases_{len(cases)}_lt_{MIN_CASES_FOR_LEARNED}",
            "weights": {"baseline_score": 1.0},
        }
        WEIGHTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_cols, indent=2, ensure_ascii=False), encoding="utf-8")
        Path("reports/reranker_cv_summary.json").write_text(
            json.dumps(
                {
                    "status": "fallback",
                    "reason": payload["fallback_reason"],
                    "case_count": len(cases),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"status": "fallback", "case_count": len(cases)}, ensure_ascii=False))
        return

    # grouped split: odd/even by case id index
    case_ids = sorted({r["case_id"] for r in all_rows})
    train_case_ids = set(case_ids[::2])
    valid_case_ids = set(case_ids[1::2])
    train_rows = [r for r in all_rows if r["case_id"] in train_case_ids]
    valid_rows = [r for r in all_rows if r["case_id"] in valid_case_ids]

    best_weights = _weight_search(train_rows)
    valid_scored = _score_with_weights(valid_rows, best_weights)
    valid_metrics = _eval_rows(valid_scored)

    payload = {
        "enabled": True,
        "version": "weight_search_v1",
        "model_type": "weight_search",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": feature_cols,
        "weights": best_weights,
    }
    WEIGHTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_cols, indent=2, ensure_ascii=False), encoding="utf-8")
    MODEL_PATH.write_text("optional learned ranker not trained in minimal mode", encoding="utf-8")

    Path("reports/reranker_cv_summary.json").write_text(
        json.dumps(
            {
                "status": "trained_weight_search",
                "case_count": len(cases),
                "train_case_count": len(train_case_ids),
                "valid_case_count": len(valid_case_ids),
                "valid_metrics": {
                    "top5_hit_rate": round(valid_metrics.top5, 6),
                    "top3_hit_rate": round(valid_metrics.top3, 6),
                    "mean_reciprocal_rank": round(valid_metrics.mrr, 6),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "trained", "weights": best_weights}, ensure_ascii=False))


if __name__ == "__main__":
    main()

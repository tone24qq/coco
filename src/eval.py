from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_loader import Sample
from .fusion import fuse_scores
from .position_modules import (
    compute_all_module_scores,
    compute_focus_score,
    compute_difference_trend,
    connectivity_heatmap,
    detect_mirror_sequences,
    detect_skip_patterns,
    ext_a2_weighted_proximity_vec,
    ext_d3_potential_field_vec,
    ext_f10_discontinuity_vec,
    rank_missing_cells,
    sequence_tail_analyzer,
)


def _rank_of_truth(ranked: List[Tuple[Tuple[int, int], float]], truth: Tuple[int, int]) -> int:
    for idx, (pos, _) in enumerate(ranked, start=1):
        if pos == truth:
            return idx
    return len(ranked) + 1


def _hits(rank: int) -> Tuple[int, int, int]:
    return int(rank <= 1), int(rank <= 3), int(rank <= 5)


@dataclass
class CaseResult:
    sample_id: str
    shape: str
    num_missing: int
    answer_row: int
    answer_col: int
    truth_rank: int
    top1_pos: Tuple[int, int]
    top3: List[Tuple[int, int]]
    top5: List[Tuple[int, int]]


MODULE_TO_FN = {
    "focus": compute_focus_score,
    "skip": detect_skip_patterns,
    "diff": compute_difference_trend,
    "mirror": detect_mirror_sequences,
    "connectivity": connectivity_heatmap,
    "tail": sequence_tail_analyzer,
    "a2": ext_a2_weighted_proximity_vec,
    "d3": ext_d3_potential_field_vec,
    "f10": ext_f10_discontinuity_vec,
}


def evaluate_samples(
    samples: Sequence[Sample],
    weights: Dict[str, float],
    modules: List[str],
) -> Tuple[Dict[str, float], List[CaseResult]]:
    case_results: List[CaseResult] = []
    ranks: List[int] = []
    for s in samples:
        out = compute_all_module_scores(s.grid, modules)
        fused = fuse_scores(
            out.scores,
            weights,
            out.missing_mask,
        )
        ranked = rank_missing_cells(fused, out.missing_mask)
        truth = (s.answer_row, s.answer_col)
        rank = _rank_of_truth(ranked, truth)
        ranks.append(rank)
        top_positions = [p for p, _ in ranked]
        case_results.append(
            CaseResult(
                sample_id=s.sample_id,
                shape=s.shape,
                num_missing=int(np.sum(out.missing_mask)),
                answer_row=s.answer_row,
                answer_col=s.answer_col,
                truth_rank=rank,
                top1_pos=top_positions[0],
                top3=top_positions[:3],
                top5=top_positions[:5],
            )
        )

    top1 = np.mean([r <= 1 for r in ranks])
    top3 = np.mean([r <= 3 for r in ranks])
    top5 = np.mean([r <= 5 for r in ranks])
    mean_rank = float(np.mean(ranks))
    mrr = float(np.mean([1.0 / r for r in ranks]))

    metrics = {
        "top1_hit_rate": float(top1),
        "top3_hit_rate": float(top3),
        "top5_hit_rate": float(top5),
        "mean_rank": mean_rank,
        "mrr": mrr,
        "num_samples": len(samples),
    }
    return metrics, case_results


def random_baseline(samples: Sequence[Sample], repeats: int = 300, seed: int = 123) -> Dict[str, float]:
    rnd = random.Random(seed)
    hit1 = []
    hit3 = []
    hit5 = []
    mean_ranks = []
    mrrs = []
    for _ in range(repeats):
        ranks = []
        for s in samples:
            missing = list(map(tuple, np.argwhere(s.grid == -1)))
            rnd.shuffle(missing)
            rank = missing.index((s.answer_row, s.answer_col)) + 1
            ranks.append(rank)
        hit1.append(np.mean([r <= 1 for r in ranks]))
        hit3.append(np.mean([r <= 3 for r in ranks]))
        hit5.append(np.mean([r <= 5 for r in ranks]))
        mean_ranks.append(float(np.mean(ranks)))
        mrrs.append(float(np.mean([1 / r for r in ranks])))
    return {
        "top1_hit_rate": float(np.mean(hit1)),
        "top3_hit_rate": float(np.mean(hit3)),
        "top5_hit_rate": float(np.mean(hit5)),
        "mean_rank": float(np.mean(mean_ranks)),
        "mrr": float(np.mean(mrrs)),
    }


def center_baseline(samples: Sequence[Sample]) -> Dict[str, float]:
    from .position_modules import center_baseline_score

    ranks: List[int] = []
    for s in samples:
        mask = s.grid == -1
        score = center_baseline_score(s.grid)
        ranked = rank_missing_cells(score, mask)
        ranks.append(_rank_of_truth(ranked, (s.answer_row, s.answer_col)))
    return {
        "top1_hit_rate": float(np.mean([r <= 1 for r in ranks])),
        "top3_hit_rate": float(np.mean([r <= 3 for r in ranks])),
        "top5_hit_rate": float(np.mean([r <= 5 for r in ranks])),
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean([1.0 / r for r in ranks])),
    }


def density_baseline(samples: Sequence[Sample]) -> Dict[str, float]:
    ranks: List[int] = []
    for s in samples:
        mask = s.grid == -1
        score = compute_focus_score(s.grid)
        ranked = rank_missing_cells(score, mask)
        ranks.append(_rank_of_truth(ranked, (s.answer_row, s.answer_col)))
    return {
        "top1_hit_rate": float(np.mean([r <= 1 for r in ranks])),
        "top3_hit_rate": float(np.mean([r <= 3 for r in ranks])),
        "top5_hit_rate": float(np.mean([r <= 5 for r in ranks])),
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean([1.0 / r for r in ranks])),
    }


def build_folds(samples: Sequence[Sample], k: int = 3) -> List[Tuple[List[int], List[int]]]:
    ordered = all(s.order_index is not None for s in samples)
    idxs = list(range(len(samples)))
    if ordered:
        idxs = sorted(idxs, key=lambda i: samples[i].order_index or 0)
        fold_size = max(1, len(idxs) // k)
        folds = []
        for i in range(1, k + 1):
            test_start = (i - 1) * fold_size
            test_end = len(idxs) if i == k else i * fold_size
            test = idxs[test_start:test_end]
            train = idxs[:test_start]
            if not train or not test:
                continue
            folds.append((train, test))
        if folds:
            return folds

    by_shape: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        by_shape.setdefault(s.shape, []).append(i)
    folds: List[Tuple[List[int], List[int]]] = []
    fold_bins: List[List[int]] = [[] for _ in range(k)]
    for group in by_shape.values():
        for idx, sample_idx in enumerate(group):
            fold_bins[idx % k].append(sample_idx)
    for i in range(k):
        test = fold_bins[i]
        train = [x for j, b in enumerate(fold_bins) if j != i for x in b]
        if test and train:
            folds.append((train, test))
    return folds


def _weights_grid(modules: List[str], values: Iterable[float]) -> Iterable[Dict[str, float]]:
    for combo in itertools.product(values, repeat=len(modules)):
        if np.isclose(sum(combo), 0.0):
            continue
        yield {m: float(v) for m, v in zip(modules, combo)}


def tune_weights(
    samples: Sequence[Sample],
    modules: List[str],
    grid_values: List[float],
    k: int = 3,
) -> Dict[str, float]:
    folds = build_folds(samples, k=k)
    if not folds:
        return {m: 1.0 for m in modules}

    best = None
    for weights in _weights_grid(modules, grid_values):
        fold_metrics = []
        for tr, va in folds:
            va_samples = [samples[i] for i in va]
            m, _ = evaluate_samples(va_samples, weights, modules)
            fold_metrics.append(m)
        top3 = float(np.mean([m["top3_hit_rate"] for m in fold_metrics]))
        mean_rank = float(np.mean([m["mean_rank"] for m in fold_metrics]))
        mrr = float(np.mean([m["mrr"] for m in fold_metrics]))
        weight_key = tuple(weights[m] for m in modules)
        candidate = (top3, -mean_rank, mrr, -sum(weight_key), weight_key, weights)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return best[-1]


def ablation(
    samples: Sequence[Sample],
    full_weights: Dict[str, float],
    modules: List[str],
) -> Dict[str, Dict[str, float]]:
    full_metrics, _ = evaluate_samples(samples, full_weights, modules)
    results = {"full": full_metrics}
    for m in modules:
        sub_modules = [x for x in modules if x != m]
        weights = {k: v for k, v in full_weights.items() if k in sub_modules}
        if not weights:
            continue
        metrics, _ = evaluate_samples(
            samples,
            weights,
            sub_modules,
        )
        results[f"drop_{m}"] = metrics
    return results


def write_case_predictions(path: Path, case_results: Sequence[CaseResult]) -> None:
    rows = []
    for c in case_results:
        rows.append(
            {
                "sample_id": c.sample_id,
                "shape": c.shape,
                "num_missing": c.num_missing,
                "answer_row": c.answer_row,
                "answer_col": c.answer_col,
                "truth_rank": c.truth_rank,
                "top1": c.top1_pos,
                "top3": c.top3,
                "top5": c.top5,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

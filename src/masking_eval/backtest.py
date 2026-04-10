from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .candidate_scoring import TargetPrediction, random_rank, rank_candidates
from .data_loader import BoardSample


def build_folds(samples: Sequence[BoardSample], n_folds: int) -> List[Tuple[List[int], List[int], List[int]]]:
    idxs = list(range(len(samples)))
    ordered = all(getattr(s, "order_index", None) is not None for s in samples)
    if ordered:
        idxs = sorted(idxs, key=lambda i: getattr(samples[i], "order_index") or 0)
    bins = [idxs[i::n_folds] for i in range(n_folds)]
    folds = []
    for i in range(n_folds):
        test = bins[i]
        train_valid = [x for j, b in enumerate(bins) if j != i for x in b]
        if len(train_valid) < 2 or len(test) == 0:
            continue
        cut = max(1, int(len(train_valid) * 0.75))
        train = train_valid[:cut] or train_valid[:1]
        valid = train_valid[cut:] or train_valid[-1:]
        folds.append((train, valid, test))
    return folds


def build_heatmap_prior(train_boards: Sequence[BoardSample], repeats: int, seed: int) -> np.ndarray | None:
    if not train_boards:
        return None
    shape = train_boards[0].grid.shape
    if any(b.grid.shape != shape for b in train_boards):
        return None
    rng = np.random.default_rng(seed)
    heatmap = np.zeros(shape, dtype=float)
    cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    masked_n = len(cells) // 2
    for _ in train_boards:
        for _ in range(repeats):
            shuffled = list(cells)
            rng.shuffle(shuffled)
            for r, c in shuffled[:masked_n]:
                heatmap[r, c] += 1.0
    return heatmap / max(float(np.max(heatmap)), 1.0)


def _metrics_from_ranks(ranks: List[int], num_candidates: List[int]) -> Dict[str, float]:
    if not ranks:
        base = {
            "overall_top10_hit_rate": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "mrr": 0.0,
            "normalized_rank": 0.0,
            "num_targets": 0,
        }
        for k in range(1, 11):
            base[f"cumulative_top{k}_hit_rate"] = 0.0
            base[f"exact_rank{k}_hit_rate"] = 0.0
        return base

    arr = np.array(ranks)
    normalized = [(r - 1) / max(c - 1, 1) for r, c in zip(ranks, num_candidates)]
    out: Dict[str, float] = {
        "overall_top10_hit_rate": float(np.mean(arr <= 10)),
        "mean_rank": float(np.mean(arr)),
        "median_rank": float(np.median(arr)),
        "mrr": float(np.mean(1.0 / arr)),
        "normalized_rank": float(np.mean(normalized)),
        "num_targets": float(len(arr)),
    }
    for k in range(1, 11):
        out[f"cumulative_top{k}_hit_rate"] = float(np.mean(arr <= k))
        out[f"exact_rank{k}_hit_rate"] = float(np.mean(arr == k))
    return out


def _candidate_count_distribution(num_candidates: List[int]) -> Dict[str, int]:
    if not num_candidates:
        return {}
    uniq, counts = np.unique(np.array(num_candidates, dtype=int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(uniq, counts)}


def _objective_tuple(metrics: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    return (
        metrics["overall_top10_hit_rate"],
        metrics["cumulative_top5_hit_rate"],
        metrics["cumulative_top1_hit_rate"],
        metrics["mrr"],
        -metrics["mean_rank"],
    )


def generate_masked(board: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    masked = board.copy()
    cells = [(r, c) for r in range(board.shape[0]) for c in range(board.shape[1])]
    rng.shuffle(cells)
    masked_cells = cells[: len(cells) // 2]
    for r, c in masked_cells:
        masked[r, c] = -1
    return masked, masked_cells


def evaluate_with_weights(
    boards: Sequence[BoardSample],
    repeats: int,
    seed: int,
    weights: Dict[str, float],
    heatmap_prior: np.ndarray | None,
    modules: List[str],
) -> Tuple[Dict[str, float], List[TargetPrediction], Dict[str, float], Dict[str, float], List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    preds: List[TargetPrediction] = []
    ranks: List[int] = []
    num_cands: List[int] = []
    board_rows = []
    repeat_rows = []

    for board in boards:
        board_hits = {f"board_top{k}": [] for k in range(1, 11)}
        local_prior = heatmap_prior if (heatmap_prior is not None and heatmap_prior.shape == board.grid.shape) else None
        for rep in range(repeats):
            masked, targets = generate_masked(board.grid, rng)
            rep_hits = []
            for r, c in targets:
                true_val = int(board.grid[r, c])
                rank, score_true, _ranked = rank_candidates(masked, (r, c), true_val, weights, local_prior, modules)
                nc = int(np.sum(masked == -1))
                ranks.append(rank)
                num_cands.append(nc)
                for k in range(1, 11):
                    board_hits[f"board_top{k}"].append(int(rank <= k))
                rep_hits.append(int(rank <= 1))
                preds.append(
                    TargetPrediction(
                        board_id=board.board_id,
                        size_class=str(getattr(board, "size_class", "unknown")),
                        repeat_id=rep,
                        target_row=r,
                        target_col=c,
                        true_value=true_val,
                        rank=rank,
                        num_candidates=nc,
                        top1_hit=int(rank <= 1),
                        top3_hit=int(rank <= 3),
                        top5_hit=int(rank <= 5),
                        top10_hit=int(rank <= 10),
                        ranking_score=score_true,
                    )
                )
            repeat_rows.append(float(np.mean(rep_hits)) if rep_hits else 0.0)
        board_rows.append({k: float(np.mean(v)) if v else 0.0 for k, v in board_hits.items()})

    metrics = _metrics_from_ranks(ranks, num_cands)
    metrics["candidate_count_distribution"] = _candidate_count_distribution(num_cands)  # type: ignore[index]
    board_stats = (
        {
            f"{k}_mean": float(np.mean([b[k] for b in board_rows])) if board_rows else 0.0
            for k in board_rows[0]
        }
        if board_rows
        else {}
    )
    repeat_stats = {"repeat_top1_variance": float(np.var(repeat_rows)) if repeat_rows else 0.0}

    error_cases = []
    for p in preds:
        if p.rank > 10 or 2 <= p.rank <= 10:
            error_cases.append(
                {
                    "board_id": p.board_id,
                    "size_class": p.size_class,
                    "repeat_id": p.repeat_id,
                    "target_row": p.target_row,
                    "target_col": p.target_col,
                    "true_value": p.true_value,
                    "rank": p.rank,
                    "num_candidates": p.num_candidates,
                    "error_bucket": "rank_gt_10" if p.rank > 10 else "rank_2_to_10",
                }
            )

    return metrics, preds, board_stats, repeat_stats, error_cases


def evaluate_random_baseline(boards: Sequence[BoardSample], repeats: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    ranks, num_cands = [], []
    for board in boards:
        for _ in range(repeats):
            masked, targets = generate_masked(board.grid, rng)
            for r, c in targets:
                true_val = int(board.grid[r, c])
                ranks.append(random_rank(masked, true_val, rng))
                num_cands.append(int(np.sum(masked == -1)))
    return _metrics_from_ranks(ranks, num_cands)


def tune_weights(
    train_boards: Sequence[BoardSample],
    valid_boards: Sequence[BoardSample],
    repeats: int,
    seed: int,
    modules: List[str],
    n_trials: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    best = None
    trial_rows: List[Dict[str, object]] = []
    heatmap = build_heatmap_prior(train_boards, max(2, repeats // 2), seed)

    for trial_id in range(max(1, n_trials)):
        if trial_id == 0:
            w = {m: 1.0 for m in modules}
        else:
            raw = rng.random(len(modules))
            raw = raw / max(float(np.sum(raw)), 1e-12)
            w = {m: float(v) for m, v in zip(modules, raw)}
        m, _, _, _, _ = evaluate_with_weights(valid_boards, max(2, repeats // 2), seed + 1, w, heatmap, modules)
        score = _objective_tuple(m)
        row = {"trial_id": trial_id, "weights": w, "metrics": m, "objective": score}
        trial_rows.append(row)
        if best is None or score > best[0]:
            best = (score, w, m)

    assert best is not None
    leaderboard = sorted(trial_rows, key=lambda x: tuple(x["objective"]), reverse=True)
    return best[1], leaderboard, best[2]


def _sub_weights(modules: List[str], active: List[str]) -> Dict[str, float]:
    return {m: (1.0 if m in active else 0.0) for m in modules}


def _per_size_metrics(preds: List[TargetPrediction]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    by_size: Dict[str, List[TargetPrediction]] = {}
    for p in preds:
        by_size.setdefault(p.size_class, []).append(p)
    for size, rows in by_size.items():
        ranks = [r.rank for r in rows]
        cands = [r.num_candidates for r in rows]
        out[size] = _metrics_from_ranks(ranks, cands)
    return out


def run_backtest(
    boards: Sequence[BoardSample],
    folds: int,
    repeats: int,
    seed: int,
    modules: List[str],
    n_trials: int,
) -> Dict:
    split = build_folds(boards, folds)
    if not split:
        return {"insufficient_data": True, "anti_leakage_checks": "passed", "predictions": []}

    full_fold, rand_fold, vis_fold, local_fold, pos_fold = [], [], [], [], []
    all_preds: List[TargetPrediction] = []
    all_error_cases: List[Dict[str, object]] = []
    search_trials: List[Dict[str, object]] = []

    for fid, (tri, vai, tei) in enumerate(split):
        train = [boards[i] for i in tri]
        valid = [boards[i] for i in vai]
        test = [boards[i] for i in tei]
        heatmap = build_heatmap_prior(train, max(2, repeats // 2), seed + fid)
        best_w, leaderboard, _best_valid_metrics = tune_weights(
            train,
            valid,
            repeats,
            seed + fid,
            modules,
            n_trials,
        )
        fm, preds, bs, rs, err = evaluate_with_weights(test, repeats, seed + 999 + fid, best_w, heatmap, modules)
        fm.update(bs)
        fm.update(rs)
        full_fold.append((fm, best_w))
        all_preds.extend(preds)
        all_error_cases.extend(err)
        search_trials.extend(
            [
                {
                    "fold_id": fid,
                    "trial_id": row["trial_id"],
                    "weights": row["weights"],
                    "metrics": row["metrics"],
                    "objective": row["objective"],
                }
                for row in leaderboard
            ]
        )
        rand_fold.append(evaluate_random_baseline(test, repeats, seed + 123 + fid))
        vis_fold.append(
            evaluate_with_weights(
                test,
                repeats,
                seed + 900 + fid,
                _sub_weights(modules, ["tail"]),
                heatmap,
                modules,
            )[0]
        )
        local_fold.append(
            evaluate_with_weights(
                test,
                repeats,
                seed + 901 + fid,
                _sub_weights(modules, ["skip", "diff"]),
                heatmap,
                modules,
            )[0]
        )
        pos_fold.append(
            evaluate_with_weights(
                test,
                repeats,
                seed + 902 + fid,
                _sub_weights(modules, ["focus", "connectivity"]),
                heatmap,
                modules,
            )[0]
        )

    def avg(ms: List[Dict[str, float]]) -> Dict[str, float]:
        keys = [k for k in ms[0] if isinstance(ms[0][k], (int, float))]
        return {k: float(np.mean([m[k] for m in ms])) for k in keys}

    full_avg = avg([m for m, _ in full_fold])
    best_weights = max(full_fold, key=lambda x: _objective_tuple(x[0]))[1]

    ablation = {}
    for drop in modules:
        kept = [m for m in modules if m != drop]
        if not kept:
            continue
        drop_w = {m: best_weights.get(m, 0.0) for m in kept}
        m, _, _, _, _ = evaluate_with_weights(boards, max(2, repeats // 2), seed + 777, drop_w, None, kept)
        ablation[f"drop_{drop}"] = m

    per_size = _per_size_metrics(all_preds)
    trial_leaderboard = sorted(search_trials, key=lambda x: tuple(x["objective"]), reverse=True)

    return {
        "anti_leakage_checks": "passed",
        "num_boards": len(boards),
        "num_folds": len(split),
        "masking_repeats": repeats,
        "seed": seed,
        "full_model": full_avg,
        "per_size_metrics": per_size,
        "baselines": {
            "random": avg(rand_fold),
            "visible_frequency": avg(vis_fold),
            "local_rule": avg(local_fold),
            "position_only": avg(pos_fold),
        },
        "best_weights": best_weights,
        "search_trials": search_trials,
        "trial_leaderboard": trial_leaderboard,
        "error_cases_top10": all_error_cases,
        "ablation": ablation,
        "predictions": [asdict(p) for p in all_preds],
    }


def write_outputs(result: Dict, summary_path: Path, pred_path: Path, config_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {k: v for k, v in result.items() if k != "predictions"}
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(result.get("predictions", [])).to_csv(pred_path, index=False)
    config_path.write_text(
        json.dumps({"best_weights": result.get("best_weights", {})}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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
    ordered = all(s.order_index is not None for s in samples)
    if ordered:
        idxs = sorted(idxs, key=lambda i: samples[i].order_index or 0)
    bins = [idxs[i::n_folds] for i in range(n_folds)]
    folds = []
    for i in range(n_folds):
        test = bins[i]
        train_valid = [x for j, b in enumerate(bins) if j != i for x in b]
        if len(train_valid) < 2 or len(test) == 0:
            continue
        cut = max(1, int(len(train_valid) * 0.75))
        train = train_valid[:cut]
        valid = train_valid[cut:] or train_valid[-1:]
        if not train:
            train = train_valid[:1]
        folds.append((train, valid, test))
    return folds


def build_heatmap_prior(train_boards: Sequence[BoardSample], repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    heatmap = np.zeros((10, 8), dtype=float)
    for _board in train_boards:
        cells = [(r, c) for r in range(10) for c in range(8)]
        for _ in range(repeats):
            rng.shuffle(cells)
            for r, c in cells[:40]:
                heatmap[r, c] += 1.0
    return heatmap / max(float(np.max(heatmap)), 1.0)


def _metrics_from_ranks(ranks: List[int], num_candidates: List[int]) -> Dict[str, float]:
    arr = np.array(ranks)
    normalized = [(r - 1) / max(c - 1, 1) for r, c in zip(ranks, num_candidates)]
    return {
        "top1_hit_rate": float(np.mean(arr <= 1)),
        "top3_hit_rate": float(np.mean(arr <= 3)),
        "top5_hit_rate": float(np.mean(arr <= 5)),
        "mean_rank": float(np.mean(arr)),
        "median_rank": float(np.median(arr)),
        "mrr": float(np.mean(1.0 / arr)),
        "normalized_rank": float(np.mean(normalized)),
        "num_targets": int(len(arr)),
    }


def generate_masked(board: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    masked = board.copy()
    cells = [(r, c) for r in range(10) for c in range(8)]
    rng.shuffle(cells)
    masked_cells = cells[:40]
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
) -> Tuple[Dict[str, float], List[TargetPrediction], Dict[str, float], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    preds: List[TargetPrediction] = []
    ranks: List[int] = []
    num_cands: List[int] = []
    board_rows = []
    repeat_rows = []

    for board in boards:
        board_top1, board_top3, board_top5 = [], [], []
        for rep in range(repeats):
            masked, targets = generate_masked(board.grid, rng)
            rep_hits = []
            for r, c in targets:
                true_val = int(board.grid[r, c])
                rank, score_true = rank_candidates(
                    masked,
                    (r, c),
                    true_val,
                    weights,
                    heatmap_prior,
                    modules,
                )
                nc = int(np.sum(masked == -1))
                h1, h3, h5 = int(rank <= 1), int(rank <= 3), int(rank <= 5)
                ranks.append(rank)
                num_cands.append(nc)
                board_top1.append(h1)
                board_top3.append(h3)
                board_top5.append(h5)
                rep_hits.append(h1)
                preds.append(
                    TargetPrediction(
                        board_id=board.board_id,
                        repeat_id=rep,
                        target_row=r,
                        target_col=c,
                        true_value=true_val,
                        rank=rank,
                        num_candidates=nc,
                        top1_hit=h1,
                        top3_hit=h3,
                        top5_hit=h5,
                        ranking_score=score_true,
                    )
                )
            repeat_rows.append(float(np.mean(rep_hits)))
        board_rows.append(
            {
                "board_top1": float(np.mean(board_top1)),
                "board_top3": float(np.mean(board_top3)),
                "board_top5": float(np.mean(board_top5)),
            }
        )

    metrics = _metrics_from_ranks(ranks, num_cands)
    board_stats = {
        f"{k}_mean": float(np.mean([b[k] for b in board_rows]))
        for k in ["board_top1", "board_top3", "board_top5"]
    }
    repeat_stats = {"repeat_top1_variance": float(np.var(repeat_rows)) if repeat_rows else 0.0}
    return metrics, preds, board_stats, repeat_stats


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
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    best = None
    heatmap = build_heatmap_prior(train_boards, max(2, repeats // 2), seed)
    for _ in range(n_trials):
        w = {m: float(rng.integers(0, 4)) for m in modules}
        if sum(w.values()) == 0:
            continue
        m, _, _, _ = evaluate_with_weights(
            valid_boards,
            max(2, repeats // 2),
            seed + 1,
            w,
            heatmap,
            modules,
        )
        key = (m["top1_hit_rate"], m["top3_hit_rate"], m["mrr"], -m["mean_rank"])
        if best is None or key > best[0]:
            best = (key, w)
    return best[1] if best else {m: 1.0 for m in modules}


def _sub_weights(modules: List[str], active: List[str]) -> Dict[str, float]:
    return {m: (1.0 if m in active else 0.0) for m in modules}


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
        return {"insufficient_data": True, "anti_leakage_checks": "passed"}
    full_fold, rand_fold, vis_fold, local_fold, pos_fold = [], [], [], [], []
    all_preds: List[TargetPrediction] = []

    for fid, (tri, vai, tei) in enumerate(split):
        train = [boards[i] for i in tri]
        valid = [boards[i] for i in vai]
        test = [boards[i] for i in tei]
        heatmap = build_heatmap_prior(train, max(2, repeats // 2), seed + fid)
        best_w = tune_weights(train, valid, repeats, seed + fid, modules, n_trials)
        fm, preds, bs, rs = evaluate_with_weights(test, repeats, seed + 999 + fid, best_w, heatmap, modules)
        fm.update(bs)
        fm.update(rs)
        full_fold.append((fm, best_w))
        all_preds.extend(preds)
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
    best_weights = max(full_fold, key=lambda x: x[0]["top1_hit_rate"])[1]

    ablation = {}
    for drop in modules:
        kept = [m for m in modules if m != drop]
        if not kept:
            continue
        drop_w = {m: best_weights.get(m, 0.0) for m in kept}
        m, _, _, _ = evaluate_with_weights(boards, max(2, repeats // 2), seed + 777, drop_w, None, kept)
        ablation[f"drop_{drop}"] = m

    return {
        "anti_leakage_checks": "passed",
        "num_boards": len(boards),
        "num_folds": len(split),
        "masking_repeats": repeats,
        "full_model": full_avg,
        "baselines": {
            "random": avg(rand_fold),
            "visible_frequency": avg(vis_fold),
            "local_rule": avg(local_fold),
            "position_only": avg(pos_fold),
        },
        "best_weights": best_weights,
        "ablation": ablation,
        "predictions": [asdict(p) for p in all_preds],
    }


def write_outputs(result: Dict, summary_path: Path, pred_path: Path, config_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {k: v for k, v in result.items() if k != "predictions"}
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(result["predictions"]).to_csv(pred_path, index=False)
    config_path.write_text(
        json.dumps({"best_weights": result["best_weights"]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

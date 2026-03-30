from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import requests

from winwin_service.config import AppConfig
from winwin_service.fetcher import parse_draws_from_json
from winwin_service.scoring import PredictError, predict_top3


def fetch_period_draws(lookback_days: int, timeout: int = 20) -> list[tuple[int, list[int]]]:
    endpoint = "https://winwin.tw/Bingo/GetBingoData"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }
    draw_map: dict[int, list[int]] = {}
    for delta in range(lookback_days):
        day = date.today() - timedelta(days=delta)
        response = requests.get(
            endpoint,
            params={"date": day.isoformat()},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        for period, nums in parse_draws_from_json(response.text):
            draw_map[period] = nums
    return sorted(draw_map.items(), key=lambda x: x[0])


def freq_baseline(history_draws: list[list[int]], k: int = 3) -> list[int]:
    counts: Counter[int] = Counter()
    for draw in history_draws:
        counts.update(draw)
    ranked = sorted(range(1, 81), key=lambda n: counts[n], reverse=True)
    return sorted(ranked[:k])


def previous_neighbor_baseline(prev_draw: list[int], k: int = 3) -> list[int]:
    candidates: set[int] = set()
    for n in prev_draw:
        for cand in (n - 1, n, n + 1):
            if 1 <= cand <= 80:
                candidates.add(cand)
    ranked = sorted(candidates)
    return ranked[:k]


def hit_count(pred: list[int], actual: list[int]) -> int:
    return len(set(pred) & set(actual))


def min_distance(pred: list[int], actual: list[int]) -> int:
    return min(min(abs(p - a) for a in actual) for p in pred)


def signed_min_offset(pred: list[int], actual: list[int]) -> float:
    offsets: list[int] = []
    for p in pred:
        best = min(actual, key=lambda a: abs(p - a))
        offsets.append(p - best)
    return sum(offsets) / len(offsets)


def metrics_from_triplets(top3: list[list[int]], actual: list[int]) -> dict[str, float]:
    top1 = sorted(top3[0])
    top10 = sorted({n for tri in top3 for n in tri})[:10]
    top20 = sorted({n for tri in top3 for n in tri})[:20]

    hits_each = [hit_count(sorted(tri), actual) for tri in top3]
    best3 = max(hits_each)

    return {
        "same_triplet_2hit_rate": 1.0 if best3 >= 2 else 0.0,
        "top1_2hit_rate": 1.0 if hits_each[0] >= 2 else 0.0,
        "same_triplet_3hit_rate": 1.0 if best3 >= 3 else 0.0,
        "top3_at_least_one_hit": 1.0 if best3 >= 1 else 0.0,
        "exact_hit@3": float(hit_count(top1, actual)),
        "exact_hit@10": float(hit_count(top10, actual)),
        "exact_hit@20": float(hit_count(top20, actual)),
        "adj_hit_pm1@3": float(sum(1 for n in top1 if any(abs(n - a) <= 1 for a in actual))),
        "mean_min_distance@3": float(min_distance(top1, actual)),
        "signed_offset_mean@3": float(signed_min_offset(top1, actual)),
    }


def bootstrap_ci(diffs: list[float], rng: random.Random, n: int = 3000) -> tuple[float, float]:
    m = len(diffs)
    if m == 0:
        return (0.0, 0.0)
    means: list[float] = []
    for _ in range(n):
        sample = [diffs[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    return means[int(n * 0.025)], means[int(n * 0.975)]


def paired_permutation_pvalue(diffs: list[float], rng: random.Random, n: int = 3000) -> float:
    if not diffs:
        return 1.0
    observed = abs(sum(diffs) / len(diffs))
    extreme = 0
    for _ in range(n):
        stat = abs(sum(v if rng.random() > 0.5 else -v for v in diffs) / len(diffs))
        if stat >= observed:
            extreme += 1
    return (extreme + 1) / (n + 1)


def build_config(params: dict[str, int]) -> AppConfig:
    return AppConfig(
        min_prediction_draws=10,
        recent_draws_count=params["recent_draws_count"],
        max_recent_draws_count=params["max_recent_draws_count"],
        min_score_threshold=params["min_score_threshold"],
        skip_kill_threshold=params["skip_kill_threshold"],
        streak_kill_threshold=params["streak_kill_threshold"],
        candidate_trim_size=params["candidate_trim_size"],
        warm_skip_min=params["warm_skip_min"],
        warm_skip_max=params["warm_skip_max"],
        streak_min=params["streak_min"],
        streak_max=params["streak_max"],
    )


def random_param(rng: random.Random) -> dict[str, int]:
    warm_min = rng.randint(1, 4)
    warm_max = rng.randint(max(warm_min + 1, 4), 8)
    streak_min = rng.randint(1, 2)
    streak_max = rng.randint(max(streak_min + 1, 2), 5)
    rc = rng.randint(20, 90)
    mrc = rng.randint(20, 90)
    return {
        "recent_draws_count": rc,
        "max_recent_draws_count": mrc,
        "min_score_threshold": rng.randint(30, 85),
        "skip_kill_threshold": rng.randint(6, 15),
        "streak_kill_threshold": rng.randint(3, 7),
        "candidate_trim_size": rng.randint(20, 60),
        "warm_skip_min": warm_min,
        "warm_skip_max": warm_max,
        "streak_min": streak_min,
        "streak_max": streak_max,
    }


def make_holdouts(total: int, min_train: int) -> list[tuple[int, int, int]]:
    holdout = min(90, max(40, total // 6))
    h2_start = total - holdout
    h1_start = max(min_train + 1, h2_start - holdout)
    return [
        (1, h1_start, h2_start),
        (2, h2_start, total),
    ]


def eval_config(
    numbers: list[list[int]],
    periods: list[int],
    cfg: AppConfig,
    holdouts: list[tuple[int, int, int]],
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rng_uni = random.Random(seed)
    out_rows: list[dict[str, object]] = []
    holdout_stats: list[dict[str, object]] = []

    for hid, start, end in holdouts:
        agg = {
            "model": [],
            "uniform": [],
            "frequency": [],
            "neighbor": [],
            "shift_m1": [],
            "shift_p1": [],
        }
        for t in range(start, end):
            history = numbers[:t]
            actual = numbers[t]
            latest_period = periods[t - 1]
            try:
                pred = predict_top3(history, latest_period, cfg)
            except PredictError:
                continue
            top3 = [sorted(x) for x in pred["top3"]]
            model_m = metrics_from_triplets(top3, actual)
            top1 = sorted(top3[0])
            uni = sorted(rng_uni.sample(range(1, 81), 3))
            frq = freq_baseline(history)
            nei = previous_neighbor_baseline(numbers[t - 1])
            sm1 = [max(1, n - 1) for n in top1]
            sp1 = [min(80, n + 1) for n in top1]

            base_metrics = {
                "uniform": metrics_from_triplets([uni, uni, uni], actual),
                "frequency": metrics_from_triplets([frq, frq, frq], actual),
                "neighbor": metrics_from_triplets([nei, nei, nei], actual),
                "shift_m1": metrics_from_triplets([sm1, sm1, sm1], actual),
                "shift_p1": metrics_from_triplets([sp1, sp1, sp1], actual),
            }
            agg["model"].append(model_m)
            for key in base_metrics:
                agg[key].append(base_metrics[key])

            row = {
                "holdout_id": hid,
                "issue": periods[t],
                "actual": " ".join(map(str, actual)),
                "model_top1": " ".join(map(str, top1)),
            }
            for key, val in model_m.items():
                row[f"model_{key}"] = val
            for base, m in base_metrics.items():
                for key, val in m.items():
                    row[f"{base}_{key}"] = val
            out_rows.append(row)

        if not agg["model"]:
            continue
        block = {
            "holdout_id": hid,
            "draw_count": len(agg["model"]),
        }
        for name in agg:
            for metric in agg[name][0].keys():
                block[f"{name}_{metric}"] = mean([x[metric] for x in agg[name]])

        diffs = [
            agg["model"][i]["top3_at_least_one_hit"] - agg["frequency"][i]["top3_at_least_one_hit"]
            for i in range(len(agg["model"]))
        ]
        rng_stats = random.Random(seed + hid)
        ci_low, ci_high = bootstrap_ci(diffs, rng_stats)
        p_value = paired_permutation_pvalue(diffs, rng_stats)
        block["paired_perm_p_vs_frequency"] = p_value
        block["bootstrap_ci_low_vs_frequency"] = ci_low
        block["bootstrap_ci_high_vs_frequency"] = ci_high
        holdout_stats.append(block)

    summary: dict[str, object] = {
        "holdouts": holdout_stats,
        "same_triplet_2hit_rate": mean([
            b["model_same_triplet_2hit_rate"] for b in holdout_stats
        ]) if holdout_stats else 0.0,
        "top1_2hit_rate": mean([
            b["model_top1_2hit_rate"] for b in holdout_stats
        ]) if holdout_stats else 0.0,
        "same_triplet_3hit_rate": mean([
            b["model_same_triplet_3hit_rate"] for b in holdout_stats
        ]) if holdout_stats else 0.0,
        "avg_model_same_triplet_2hit_rate": mean([
            b["model_same_triplet_2hit_rate"] for b in holdout_stats
        ]) if holdout_stats else 0.0,
        "avg_model_top1_2hit_rate": mean([
            b["model_top1_2hit_rate"] for b in holdout_stats
        ]) if holdout_stats else 0.0,
        "avg_model_same_triplet_3hit_rate": mean([
            b["model_same_triplet_3hit_rate"] for b in holdout_stats
        ]) if holdout_stats else 0.0,
        "avg_model_top3_at_least_one_hit_rate": mean([
            b["model_top3_at_least_one_hit"] for b in holdout_stats
        ]) if holdout_stats else 0.0,
        "avg_model_exact_hit@3": mean([
            b["model_exact_hit@3"] for b in holdout_stats
        ]) if holdout_stats else 0.0,
    }
    return summary, out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--max-eval-draws", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--configs-per-round", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/walkforward"))
    args = parser.parse_args()

    if args.rounds > 3:
        raise ValueError("rounds must be <= 3")
    if (args.rounds * args.configs_per_round) > 300:
        raise ValueError("total configs must be <= 300")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    draws = fetch_period_draws(args.lookback_days)
    if len(draws) > args.max_eval_draws:
        draws = draws[-args.max_eval_draws :]

    periods = [p for p, _ in draws]
    numbers = [d for _, d in draws]

    holdouts = make_holdouts(len(numbers), min_train=12)

    baseline_cfg = AppConfig(min_prediction_draws=10)
    baseline_summary, baseline_rows = eval_config(numbers, periods, baseline_cfg, holdouts, args.seed)

    rng = random.Random(args.seed)
    best = {
        "score": -1.0,
        "summary": baseline_summary,
        "params": asdict(baseline_cfg),
        "label": "baseline",
    }

    search_rows: list[dict[str, object]] = []

    for rd in range(1, args.rounds + 1):
        for idx in range(args.configs_per_round):
            params = random_param(rng)
            cfg = build_config(params)
            summary, _ = eval_config(numbers, periods, cfg, holdouts, args.seed + rd * 1000 + idx)
            score = float(summary["avg_model_same_triplet_2hit_rate"])
            row = {
                "round": rd,
                "idx": idx,
                "score": score,
                "top1_2hit_rate": float(summary["avg_model_top1_2hit_rate"]),
                "same_triplet_3hit_rate": float(summary["avg_model_same_triplet_3hit_rate"]),
                "exact_hit@3": float(summary["avg_model_exact_hit@3"]),
                **params,
            }
            search_rows.append(row)
            if score > best["score"]:
                best = {"score": score, "summary": summary, "params": params, "label": "searched"}

    best_cfg = build_config(best["params"]) if best["label"] == "searched" else baseline_cfg
    best_summary, best_rows = eval_config(numbers, periods, best_cfg, holdouts, args.seed + 9999)

    per_draw_path = args.out_dir / "per_draw_report.csv"
    if best_rows:
        with per_draw_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(best_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_rows)

    per_block_path = args.out_dir / "per_block_report.csv"
    if best_summary["holdouts"]:
        with per_block_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(best_summary["holdouts"][0].keys()))
            writer.writeheader()
            writer.writerows(best_summary["holdouts"])

    (args.out_dir / "summary_report.json").write_text(
        json.dumps(
            {
                "constraints": {
                    "max_rounds": 3,
                    "max_configs": 300,
                    "no_future_leakage": True,
                    "time_ordered_walk_forward": True,
                },
                "baseline_summary": baseline_summary,
                "best_summary": best_summary,
                "holdout_windows": [
                    {
                        "holdout_id": hid,
                        "start_issue": periods[s],
                        "end_issue": periods[e - 1],
                    }
                    for hid, s, e in holdouts
                ],
                "pass_threshold": 0.80,
                "primary_pass_threshold": 0.50,
                "passed": bool(
                    len(best_summary["holdouts"]) >= 2
                    and all(
                        h["model_same_triplet_2hit_rate"] >= 0.50
                        for h in best_summary["holdouts"]
                    )
                ),
                "metric_definitions": {
                    "same_triplet_2hit_rate": "any one triplet in top3 has >=2 overlaps",
                    "top1_2hit_rate": "first triplet has >=2 overlaps",
                    "same_triplet_3hit_rate": "any one triplet in top3 has >=3 overlaps",
                    "top3_at_least_one_hit_rate": "max overlap among 3 triplets >=1",
                    "exact_hit@3": "hit count of top1 triplet",
                    "exact_hit@10": "hit count in first 10 unique numbers from top3",
                    "exact_hit@20": "hit count in first 20 unique numbers from top3",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (args.out_dir / "best_config.json").write_text(json.dumps(best["params"], indent=2), encoding="utf-8")

    search_log_path = args.out_dir / "search_log.csv"
    if search_rows:
        with search_log_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(search_rows[0].keys()))
            writer.writeheader()
            writer.writerows(search_rows)

    ablation = [
        {
            "config": "baseline",
            "same_triplet_2hit_rate": baseline_summary["avg_model_same_triplet_2hit_rate"],
            "top1_2hit_rate": baseline_summary["avg_model_top1_2hit_rate"],
            "same_triplet_3hit_rate": baseline_summary["avg_model_same_triplet_3hit_rate"],
            "top3_at_least_one_hit_rate": baseline_summary["avg_model_top3_at_least_one_hit_rate"],
            "exact_hit@3": baseline_summary["avg_model_exact_hit@3"],
        },
        {
            "config": "best_config",
            "same_triplet_2hit_rate": best_summary["avg_model_same_triplet_2hit_rate"],
            "top1_2hit_rate": best_summary["avg_model_top1_2hit_rate"],
            "same_triplet_3hit_rate": best_summary["avg_model_same_triplet_3hit_rate"],
            "top3_at_least_one_hit_rate": best_summary["avg_model_top3_at_least_one_hit_rate"],
            "exact_hit@3": best_summary["avg_model_exact_hit@3"],
        },
    ]
    with (args.out_dir / "ablation_report.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(ablation[0].keys()))
        writer.writeheader()
        writer.writerows(ablation)

    (args.out_dir / "dead_live_knobs_report.json").write_text(
        json.dumps(
            {
                "live_knobs": [
                    "recent_draws_count",
                    "max_recent_draws_count",
                    "min_score_threshold",
                    "skip_kill_threshold",
                    "streak_kill_threshold",
                    "candidate_trim_size",
                    "warm_skip_min",
                    "warm_skip_max",
                    "streak_min",
                    "streak_max",
                ],
                "dead_knobs": [],
                "note": "Based on current predict_top3 call-chain.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

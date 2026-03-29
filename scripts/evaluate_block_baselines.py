from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median

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
    draws = sorted(draw_map.items(), key=lambda x: x[0])
    return draws


def hit_count(pred: list[int], actual: list[int]) -> int:
    return len(set(pred) & set(actual))


def freq_baseline(history_draws: list[list[int]], k: int = 3) -> list[int]:
    counts: Counter[int] = Counter()
    for draw in history_draws:
        counts.update(draw)
    ranked = sorted(range(1, 81), key=lambda n: counts[n], reverse=True)
    return sorted(ranked[:k])


def neighbor_baseline(prev_draw: list[int], rng: random.Random, k: int = 3) -> list[int]:
    candidates: set[int] = set()
    for n in prev_draw:
        for cand in (n - 1, n, n + 1):
            if 1 <= cand <= 80:
                candidates.add(cand)
    if len(candidates) < k:
        candidates = set(range(1, 81))
    return sorted(rng.sample(sorted(candidates), k))


def block_metrics(hits: list[int], block_size: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for i in range(0, len(hits), block_size):
        block = hits[i : i + block_size]
        if len(block) < block_size:
            continue
        rows.append(
            {
                "total_hits": float(sum(block)),
                "at_least_one": float(sum(1 for h in block if h >= 1)),
                "exact_2plus": float(sum(1 for h in block if h >= 2)),
                "avg_hits_per_draw": float(sum(block) / block_size),
            }
        )
    return rows


def bootstrap_ci(values: list[float], rng: random.Random, n: int = 4000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    samples: list[float] = []
    m = len(values)
    for _ in range(n):
        pick = [values[rng.randrange(m)] for _ in range(m)]
        samples.append(sum(pick) / m)
    samples.sort()
    low = samples[int(0.025 * n)]
    high = samples[int(0.975 * n)]
    return (low, high)


def permutation_pvalue(values: list[float], rng: random.Random, n: int = 4000) -> float:
    if not values:
        return 1.0
    observed = abs(sum(values) / len(values))
    count = 0
    for _ in range(n):
        perm = [v if rng.random() > 0.5 else -v for v in values]
        stat = abs(sum(perm) / len(perm))
        if stat >= observed:
            count += 1
    return (count + 1) / (n + 1)


def cohen_d(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return mu / sd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-eval-draws", type=int, default=140)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    draws = fetch_period_draws(args.lookback_days)
    if len(draws) < 30:
        raise RuntimeError(f"not enough historical draws: {len(draws)}")

    if len(draws) > args.max_eval_draws:
        draws = draws[-args.max_eval_draws :]

    periods = [p for p, _ in draws]
    numbers = [d for _, d in draws]
    cfg = AppConfig(
        min_prediction_draws=10,
        max_recent_draws_count=80,
    )

    rng_random = random.Random(args.seed)
    rng_uniform = random.Random(args.seed + 1)
    rng_neighbor = random.Random(args.seed + 2)

    rows: list[dict[str, object]] = []
    orig_hits: list[int] = []
    rand_hits: list[int] = []
    uni_hits: list[int] = []
    nei_hits: list[int] = []
    freq_hits: list[int] = []

    for t in range(cfg.min_prediction_draws, len(numbers)):
        history = numbers[:t]
        latest_period = periods[t - 1]
        actual = numbers[t]

        try:
            pred = predict_top3(history, latest_period, config=cfg)
        except PredictError:
            continue

        orig = sorted(pred["top3"][0])
        rnd = sorted(rng_random.sample(range(1, 81), 3))
        uni = sorted(rng_uniform.sample(range(1, 81), 3))
        nei = neighbor_baseline(numbers[t - 1], rng_neighbor, 3)
        frq = freq_baseline(history, 3)

        oh = hit_count(orig, actual)
        rh = hit_count(rnd, actual)
        uh = hit_count(uni, actual)
        nh = hit_count(nei, actual)
        fh = hit_count(frq, actual)

        orig_hits.append(oh)
        rand_hits.append(rh)
        uni_hits.append(uh)
        nei_hits.append(nh)
        freq_hits.append(fh)

        rows.append(
            {
                "issue": periods[t],
                "origin_hit": oh,
                "random_hit": rh,
                "uniform_hit": uh,
                "neighbor_hit": nh,
                "frequency_hit": fh,
            }
        )

    block_rows: list[dict[str, object]] = []
    block_orig = block_metrics(orig_hits, args.block_size)
    block_rand = block_metrics(rand_hits, args.block_size)
    for i, (bo, br) in enumerate(zip(block_orig, block_rand), start=1):
        start_idx = (i - 1) * args.block_size + cfg.min_prediction_draws
        end_idx = start_idx + args.block_size - 1
        diff = bo["total_hits"] - br["total_hits"]
        block_rows.append(
            {
                "block_id": i,
                "start_issue": periods[start_idx],
                "end_issue": periods[end_idx],
                "original_top3_total_hits_10": bo["total_hits"],
                "baseline_top3_total_hits_10": br["total_hits"],
                "original_at_least_one_hit_count_10": bo["at_least_one"],
                "baseline_at_least_one_hit_count_10": br["at_least_one"],
                "hit_diff_10": diff,
                "winner": "original" if diff > 0 else "baseline" if diff < 0 else "tie",
            }
        )

    diff_values = [float(r["hit_diff_10"]) for r in block_rows]
    rng_stats = random.Random(args.seed + 99)
    ci = bootstrap_ci(diff_values, rng_stats)
    p_value = permutation_pvalue(diff_values, rng_stats)

    summary = {
        "total_blocks": len(block_rows),
        "total_draws_evaluated": len(rows),
        "original_mean_hits_10": mean([r["original_top3_total_hits_10"] for r in block_rows]) if block_rows else 0.0,
        "baseline_mean_hits_10": mean([r["baseline_top3_total_hits_10"] for r in block_rows]) if block_rows else 0.0,
        "mean_diff": mean(diff_values) if diff_values else 0.0,
        "median_diff": median(diff_values) if diff_values else 0.0,
        "original_win_rate": (
            sum(1 for r in block_rows if r["winner"] == "original") / len(block_rows)
            if block_rows
            else 0.0
        ),
        "confidence_interval": {"low": ci[0], "high": ci[1]},
        "p_value": p_value,
        "effect_size": cohen_d(diff_values),
        "single_draw_hit_distribution": {
            "original": {str(k): orig_hits.count(k) for k in range(4)},
            "random": {str(k): rand_hits.count(k) for k in range(4)},
        },
        "random_expected_hits_per_draw": 0.75,
        "random_expected_hits_per_10_draws": 7.5,
    }

    per_draw_path = args.out_dir / "per_draw_results.csv"
    with per_draw_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    block_path = args.out_dir / "block_comparison.csv"
    with block_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(block_rows[0].keys()))
        writer.writeheader()
        writer.writerows(block_rows)

    dist_path = args.out_dir / "block_distribution.csv"
    with dist_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["hit_diff_10", "count"])
        writer.writeheader()
        dist = Counter(diff_values)
        for key in sorted(dist):
            writer.writerow({"hit_diff_10": key, "count": dist[key]})

    summary_json = args.out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    conclusion = (
        "original_above_random_stably"
        if summary["mean_diff"] > 0 and summary["confidence_interval"]["low"] > 0
        else "no_stable_edge_over_random"
    )
    summary_md = args.out_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Block Evaluation Summary",
                f"- total_blocks: {summary['total_blocks']}",
                f"- original_mean_hits_10: {summary['original_mean_hits_10']:.3f}",
                f"- baseline_mean_hits_10: {summary['baseline_mean_hits_10']:.3f}",
                f"- mean_diff: {summary['mean_diff']:.3f}",
                f"- median_diff: {summary['median_diff']:.3f}",
                f"- original_win_rate: {summary['original_win_rate']:.3f}",
                "- confidence_interval: "
                f"[{summary['confidence_interval']['low']:.3f}, "
                f"{summary['confidence_interval']['high']:.3f}]",
                f"- p_value: {summary['p_value']:.4f}",
                f"- effect_size: {summary['effect_size']:.3f}",
                f"- conclusion: {conclusion}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

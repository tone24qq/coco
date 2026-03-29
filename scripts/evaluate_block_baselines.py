from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import median

import requests

from winwin_service.config import AppConfig
from winwin_service.fetcher import parse_draws_from_json
from winwin_service.scoring import PredictError, predict_top3

OUTPUT_DIR = Path("analysis_outputs")
BLOCK_SIZE = 10
SEED = 20260329
LOOKBACK_DAYS = 25
MONTE_CARLO_RUNS = 1000
MAX_EVAL_DRAWS = 120


@dataclass
class DrawRecord:
    period: int
    numbers: list[int]


def fetch_history(lookback_days: int = LOOKBACK_DAYS) -> list[DrawRecord]:
    cfg = AppConfig()
    endpoint = f"{cfg.source_url.rstrip('/')}/GetBingoData"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.google.com/",
    }

    draw_map: dict[int, list[int]] = {}
    for delta in range(lookback_days):
        day = date.today() - timedelta(days=delta)
        response = requests.get(
            endpoint,
            params={"date": day.isoformat()},
            headers=headers,
            timeout=cfg.request_timeout,
        )
        response.raise_for_status()
        for period, numbers in parse_draws_from_json(response.text):
            draw_map[period] = numbers

    rows = [DrawRecord(period=p, numbers=n) for p, n in sorted(draw_map.items())]
    if len(rows) < (cfg.min_prediction_draws + BLOCK_SIZE):
        raise RuntimeError(f"Not enough rows for evaluation: {len(rows)}")
    return rows


def count_hits(prediction: list[int], actual: list[int]) -> int:
    return len(set(prediction) & set(actual))


def random_baseline(rng: random.Random) -> list[int]:
    return sorted(rng.sample(range(1, 81), 3))


def neighbor_baseline(prev_draw: list[int], rng: random.Random) -> list[int]:
    pool: set[int] = set()
    for num in prev_draw:
        for offset in (-1, 0, 1):
            cand = num + offset
            if 1 <= cand <= 80:
                pool.add(cand)
    if len(pool) < 3:
        return random_baseline(rng)
    return sorted(rng.sample(sorted(pool), 3))


def frequency_baseline(history: list[list[int]], window: int = 30) -> list[int]:
    hist = history[-window:] if len(history) >= window else history
    counter: Counter[int] = Counter()
    for draw in hist:
        counter.update(draw)
    ranked = sorted(counter.items(), key=lambda x: (x[1], -x[0]), reverse=True)
    top = [n for n, _ in ranked[:3]]
    if len(top) < 3:
        for n in range(1, 81):
            if n not in top:
                top.append(n)
            if len(top) == 3:
                break
    return sorted(top)


def bootstrap_ci(values: list[float], runs: int = 5000) -> tuple[float, float]:
    rng = random.Random(SEED + 1)
    if not values:
        return 0.0, 0.0
    samples = []
    for _ in range(runs):
        boot = [rng.choice(values) for _ in values]
        samples.append(sum(boot) / len(boot))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return lo, hi


def permutation_p_value(values: list[float], runs: int = MONTE_CARLO_RUNS) -> float:
    rng = random.Random(SEED + 2)
    if not values:
        return 1.0
    observed = abs(sum(values) / len(values))
    ge = 0
    for _ in range(runs):
        flipped = [v if rng.random() > 0.5 else -v for v in values]
        stat = abs(sum(flipped) / len(flipped))
        if stat >= observed:
            ge += 1
    return (ge + 1) / (runs + 1)


def evaluate() -> None:
    rng = random.Random(SEED)
    rows = fetch_history()
    cfg = AppConfig(
        min_prediction_draws=10,
        max_recent_draws_count=50,
    )

    per_draw: list[dict[str, object]] = []

    start_idx = max(cfg.min_prediction_draws, len(rows) - MAX_EVAL_DRAWS)
    for idx in range(start_idx, len(rows)):
        history_rows = rows[:idx]
        target = rows[idx]
        history_draws = [r.numbers for r in history_rows]

        try:
            pred_result = predict_top3(
                history_draws,
                latest_period=history_rows[-1].period,
                config=cfg,
                include_regime_debug=False,
            )
        except PredictError:
            continue

        # Use rank1 triplet as model action for fair 3-number comparison.
        original_pick = pred_result["top3"][0]
        random_pick = random_baseline(rng)
        uniform_pick = random_baseline(rng)
        neighbor_pick = neighbor_baseline(history_rows[-1].numbers, rng)
        freq_pick = frequency_baseline(history_draws)

        per_draw.append(
            {
                "period": target.period,
                "original_pick": original_pick,
                "random_pick": random_pick,
                "uniform_pick": uniform_pick,
                "neighbor_pick": neighbor_pick,
                "frequency_pick": freq_pick,
                "original_hits": count_hits(original_pick, target.numbers),
                "random_hits": count_hits(random_pick, target.numbers),
                "uniform_hits": count_hits(uniform_pick, target.numbers),
                "neighbor_hits": count_hits(neighbor_pick, target.numbers),
                "frequency_hits": count_hits(freq_pick, target.numbers),
            }
        )

    # block comparison against random baseline
    block_rows: list[dict[str, object]] = []
    for block_id, start in enumerate(range(0, len(per_draw), BLOCK_SIZE), start=1):
        block = per_draw[start : start + BLOCK_SIZE]
        if len(block) < BLOCK_SIZE:
            break

        original_hits = sum(int(r["original_hits"]) for r in block)
        baseline_hits = sum(int(r["random_hits"]) for r in block)
        original_one = sum(1 for r in block if int(r["original_hits"]) >= 1)
        baseline_one = sum(1 for r in block if int(r["random_hits"]) >= 1)
        original_two = sum(1 for r in block if int(r["original_hits"]) >= 2)

        diff = original_hits - baseline_hits
        winner = "original" if diff > 0 else "baseline" if diff < 0 else "tie"

        block_rows.append(
            {
                "block_id": block_id,
                "start_issue": block[0]["period"],
                "end_issue": block[-1]["period"],
                "original_top3_total_hits_10": original_hits,
                "baseline_top3_total_hits_10": baseline_hits,
                "original_at_least_one_hit_count_10": original_one,
                "baseline_at_least_one_hit_count_10": baseline_one,
                "top3_exact_2plus_count_10": original_two,
                "avg_hits_per_draw": round(original_hits / BLOCK_SIZE, 4),
                "hit_diff_10": diff,
                "winner": winner,
            }
        )

    diffs = [float(r["hit_diff_10"]) for r in block_rows]
    orig_hits_blocks = [float(r["original_top3_total_hits_10"]) for r in block_rows]
    base_hits_blocks = [float(r["baseline_top3_total_hits_10"]) for r in block_rows]

    mean_diff = sum(diffs) / len(diffs)
    ci_lo, ci_hi = bootstrap_ci(diffs)
    p_value = permutation_p_value(diffs)
    std_diff = math.sqrt(
        sum((d - mean_diff) ** 2 for d in diffs) / max(1, len(diffs) - 1)
    )
    effect_size = 0.0 if std_diff == 0 else mean_diff / std_diff
    win_rate = sum(1 for d in diffs if d > 0) / len(diffs)

    draw_hit_counter = Counter(int(r["original_hits"]) for r in per_draw)

    summary = {
        "total_blocks": len(block_rows),
        "total_draws_evaluated": len(per_draw),
        "original_mean_hits_10": sum(orig_hits_blocks) / len(orig_hits_blocks),
        "baseline_mean_hits_10": sum(base_hits_blocks) / len(base_hits_blocks),
        "mean_diff": mean_diff,
        "median_diff": median(diffs),
        "original_win_rate": win_rate,
        "confidence_interval": [ci_lo, ci_hi],
        "p_value": p_value,
        "effect_size": effect_size,
        "single_draw_hit_distribution": {
            "0_hit": draw_hit_counter[0],
            "1_hit": draw_hit_counter[1],
            "2_hit": draw_hit_counter[2],
            "3_hit": draw_hit_counter[3],
        },
        "edge_blocks": [r for r in block_rows if int(r["hit_diff_10"]) > 0],
        "weak_blocks": [r for r in block_rows if int(r["hit_diff_10"]) <= 0],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "per_draw_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_draw[0].keys()))
        writer.writeheader()
        writer.writerows(per_draw)

    with (OUTPUT_DIR / "block_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(block_rows[0].keys()))
        writer.writeheader()
        writer.writerows(block_rows)

    with (OUTPUT_DIR / "block_distribution.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "mean",
                "median",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "metric": "hit_diff_10",
                "mean": round(mean_diff, 4),
                "median": round(median(diffs), 4),
                "min": min(diffs),
                "max": max(diffs),
            }
        )

    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    conclusion = (
        "Original model shows stable lift over random baseline"
        if (mean_diff > 0 and p_value < 0.05 and ci_lo > 0)
        else "No stable statistical edge over random baseline"
    )

    with (OUTPUT_DIR / "summary.md").open("w") as f:
        f.write("# Block Evaluation Summary\n\n")
        f.write(f"- total_blocks: {summary['total_blocks']}\n")
        f.write(f"- original_mean_hits_10: {summary['original_mean_hits_10']:.4f}\n")
        f.write(f"- baseline_mean_hits_10: {summary['baseline_mean_hits_10']:.4f}\n")
        f.write(f"- mean_diff: {summary['mean_diff']:.4f}\n")
        f.write(f"- median_diff: {summary['median_diff']:.4f}\n")
        f.write(f"- original_win_rate: {summary['original_win_rate']:.4f}\n")
        f.write(
            "- confidence_interval: "
            f"[{summary['confidence_interval'][0]:.4f}, "
            f"{summary['confidence_interval'][1]:.4f}]\n"
        )
        f.write(f"- p_value: {summary['p_value']:.6f}\n")
        f.write(f"- effect_size: {summary['effect_size']:.4f}\n")
        f.write(f"- conclusion: {conclusion}\n")


if __name__ == "__main__":
    evaluate()

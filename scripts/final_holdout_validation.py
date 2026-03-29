from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import requests

from winwin_service.config import AppConfig
from winwin_service.fetcher import parse_draws_from_json
from winwin_service.scoring import PredictError, predict_top3


REQUIRED_SUMMARY_KEYS = {
    "snapshot_source",
    "snapshot_fingerprint",
    "search_issue_range",
    "validation_issue_range",
    "final_holdout_issue_range",
    "final_holdout_blocks",
    "total_draws_search",
    "total_draws_validation",
    "total_draws_final_holdout",
    "chosen_config",
    "final_metrics",
    "baseline_metrics",
    "block_metrics",
    "p_value_vs_frequency",
    "bootstrap_ci_vs_frequency",
    "leakage_check_passed",
    "passed",
    "pass_reason",
}


def fetch_live_draws(lookback_days: int, timeout: int = 20) -> list[tuple[int, list[int]]]:
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


def snapshot_fingerprint(draws: list[tuple[int, list[int]]]) -> str:
    payload = json.dumps(draws, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_snapshot(path: Path, draws: list[tuple[int, list[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "draws": draws,
        "fingerprint": snapshot_fingerprint(draws),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_snapshot(path: Path) -> list[tuple[int, list[int]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    draws = [(int(p), [int(n) for n in nums]) for p, nums in raw["draws"]]
    expected = raw.get("fingerprint", "")
    got = snapshot_fingerprint(draws)
    if expected and expected != got:
        raise ValueError("snapshot fingerprint mismatch")
    return draws


def split_windows(
    total_draws: int,
    search_ratio: float,
    validation_ratio: float,
    final_ratio: float,
    min_train_draws: int,
) -> dict[str, tuple[int, int]]:
    if abs((search_ratio + validation_ratio + final_ratio) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    search_len = int(total_draws * search_ratio)
    validation_len = int(total_draws * validation_ratio)
    final_len = total_draws - search_len - validation_len
    if search_len < min_train_draws or validation_len < 8 or final_len < 16:
        raise ValueError("insufficient draws for search/validation/final splits")
    return {
        "search": (0, search_len),
        "validation": (search_len, search_len + validation_len),
        "final": (search_len + validation_len, total_draws),
    }


def split_final_blocks(final_start: int, final_end: int) -> list[tuple[int, int]]:
    size = final_end - final_start
    if size < 16:
        raise ValueError("final holdout too short for 2 blocks")
    mid = final_start + size // 2
    return [(final_start, mid), (mid, final_end)]


def leakage_check(windows: dict[str, tuple[int, int]], final_blocks: list[tuple[int, int]]) -> bool:
    s0, s1 = windows["search"]
    v0, v1 = windows["validation"]
    f0, f1 = windows["final"]
    b0s, b0e = final_blocks[0]
    b1s, b1e = final_blocks[1]
    return s0 < s1 <= v0 < v1 <= f0 == b0s < b0e <= b1s < b1e == f1


def hit_count(pred: list[int], actual: list[int]) -> int:
    return len(set(pred) & set(actual))


def signed_offsets(pred: list[int], actual: list[int]) -> list[int]:
    result: list[int] = []
    for p in pred:
        nearest = min(actual, key=lambda a: abs(p - a))
        result.append(p - nearest)
    return result


def min_distance(pred: list[int], actual: list[int]) -> int:
    return min(min(abs(p - a) for a in actual) for p in pred)


def metrics_from_top3(top3: list[list[int]], actual: list[int]) -> dict[str, float]:
    top1 = sorted(top3[0])
    unique = sorted({n for tri in top3 for n in tri})
    top10 = unique[:10]
    top20 = unique[:20]
    hits_by_triplet = [hit_count(sorted(t), actual) for t in top3]
    offsets = signed_offsets(top1, actual)

    strict_adj = sum(
        1 for n in top1 if any(abs(n - a) == 1 for a in actual) and n not in actual
    )
    overshoot = sum(1 for x in offsets if x > 0) / len(offsets)
    undershoot = sum(1 for x in offsets if x < 0) / len(offsets)

    return {
        "top3_at_least_one_hit_rate": 1.0 if max(hits_by_triplet) >= 1 else 0.0,
        "exact_hit@3": float(hit_count(top1, actual)),
        "exact_hit@10": float(hit_count(top10, actual)),
        "exact_hit@20": float(hit_count(top20, actual)),
        "adj_hit_pm1@3": float(sum(1 for n in top1 if any(abs(n - a) <= 1 for a in actual))),
        "strict_adj_only_pm1@3": float(strict_adj),
        "mean_min_distance@3": float(min_distance(top1, actual)),
        "signed_offset_mean@3": float(sum(offsets) / len(offsets)),
        "overshoot_rate@3": float(overshoot),
        "undershoot_rate@3": float(undershoot),
    }


def freq_baseline(history_draws: list[list[int]], k: int = 3) -> list[int]:
    counts: Counter[int] = Counter()
    for draw in history_draws:
        counts.update(draw)
    ranked = sorted(range(1, 81), key=lambda n: counts[n], reverse=True)
    return sorted(ranked[:k])


def previous_neighbor_baseline(prev_draw: list[int], k: int = 3) -> list[int]:
    pool: set[int] = set()
    for n in prev_draw:
        for c in (n - 1, n, n + 1):
            if 1 <= c <= 80:
                pool.add(c)
    return sorted(pool)[:k]


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


def sample_params(rng: random.Random) -> dict[str, int]:
    warm_min = rng.randint(1, 4)
    warm_max = rng.randint(max(4, warm_min + 1), 8)
    streak_min = rng.randint(1, 2)
    streak_max = rng.randint(max(2, streak_min + 1), 5)
    return {
        "recent_draws_count": rng.randint(20, 90),
        "max_recent_draws_count": rng.randint(20, 90),
        "min_score_threshold": rng.randint(35, 85),
        "skip_kill_threshold": rng.randint(6, 15),
        "streak_kill_threshold": rng.randint(3, 7),
        "candidate_trim_size": rng.randint(20, 60),
        "warm_skip_min": warm_min,
        "warm_skip_max": warm_max,
        "streak_min": streak_min,
        "streak_max": streak_max,
    }


def refine_params(base: dict[str, int], rng: random.Random, scale: int) -> dict[str, int]:
    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    warm_min = clamp(base["warm_skip_min"] + rng.randint(-1, 1), 1, 6)
    warm_max = clamp(base["warm_skip_max"] + rng.randint(-scale, scale), warm_min + 1, 9)
    streak_min = clamp(base["streak_min"] + rng.randint(-1, 1), 1, 3)
    streak_max = clamp(base["streak_max"] + rng.randint(-scale, scale), streak_min + 1, 6)

    return {
        "recent_draws_count": clamp(base["recent_draws_count"] + rng.randint(-scale * 4, scale * 4), 15, 100),
        "max_recent_draws_count": clamp(base["max_recent_draws_count"] + rng.randint(-scale * 4, scale * 4), 15, 100),
        "min_score_threshold": clamp(base["min_score_threshold"] + rng.randint(-scale * 3, scale * 3), 20, 95),
        "skip_kill_threshold": clamp(base["skip_kill_threshold"] + rng.randint(-scale, scale), 4, 20),
        "streak_kill_threshold": clamp(base["streak_kill_threshold"] + rng.randint(-1, 1), 2, 8),
        "candidate_trim_size": clamp(base["candidate_trim_size"] + rng.randint(-scale * 3, scale * 3), 10, 70),
        "warm_skip_min": warm_min,
        "warm_skip_max": warm_max,
        "streak_min": streak_min,
        "streak_max": streak_max,
    }


def bootstrap_ci(diffs: list[float], rng: random.Random, n: int = 4000) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    m = len(diffs)
    means: list[float] = []
    for _ in range(n):
        sample = [diffs[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def paired_permutation_pvalue(diffs: list[float], rng: random.Random, n: int = 4000) -> float:
    if not diffs:
        return 1.0
    observed = abs(sum(diffs) / len(diffs))
    count = 0
    for _ in range(n):
        stat = abs(sum(v if rng.random() > 0.5 else -v for v in diffs) / len(diffs))
        if stat >= observed:
            count += 1
    return (count + 1) / (n + 1)


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: mean([r[k] for r in rows]) for k in rows[0].keys()} if rows else {}


def evaluate_window(
    numbers: list[list[int]],
    periods: list[int],
    start: int,
    end: int,
    cfg: AppConfig,
    seed: int,
    include_baselines: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, float]]]]:
    rng_uni = random.Random(seed)
    out_rows: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, float]]] = {"model": []}
    if include_baselines:
        buckets.update(
            {
                "uniform_random": [],
                "frequency": [],
                "previous_neighbor": [],
                "shift_m1": [],
                "shift_p1": [],
            }
        )

    for t in range(max(start, cfg.min_prediction_draws), end):
        history = numbers[:t]
        actual = numbers[t]
        latest_period = periods[t - 1]
        pred = predict_top3(history, latest_period, cfg)
        top3 = [sorted(x) for x in pred["top3"]]

        model_metrics = metrics_from_top3(top3, actual)
        buckets["model"].append(model_metrics)

        row: dict[str, Any] = {
            "issue": periods[t],
            "latest_period": latest_period,
            "model_top3": json.dumps(top3, ensure_ascii=False),
            "actual": json.dumps(actual, ensure_ascii=False),
        }
        for k, v in model_metrics.items():
            row[f"model_{k}"] = v

        if include_baselines:
            top1 = sorted(top3[0])
            uni = sorted(rng_uni.sample(range(1, 81), 3))
            frq = freq_baseline(history)
            nei = previous_neighbor_baseline(numbers[t - 1])
            m1 = [max(1, n - 1) for n in top1]
            p1 = [min(80, n + 1) for n in top1]
            baselines = {
                "uniform_random": [uni, uni, uni],
                "frequency": [frq, frq, frq],
                "previous_neighbor": [nei, nei, nei],
                "shift_m1": [m1, m1, m1],
                "shift_p1": [p1, p1, p1],
            }
            for name, tri in baselines.items():
                metrics = metrics_from_top3(tri, actual)
                buckets[name].append(metrics)
                for k, v in metrics.items():
                    row[f"{name}_{k}"] = v

        out_rows.append(row)

    return out_rows, buckets


def candidate_key(params: dict[str, int]) -> tuple[int, ...]:
    return (
        params["recent_draws_count"],
        params["max_recent_draws_count"],
        params["min_score_threshold"],
        params["skip_kill_threshold"],
        params["streak_kill_threshold"],
        params["candidate_trim_size"],
        params["warm_skip_min"],
        params["warm_skip_max"],
        params["streak_min"],
        params["streak_max"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--max-eval-draws", type=int, default=160)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--coarse", type=int, default=90)
    parser.add_argument("--fine", type=int, default=45)
    parser.add_argument("--local", type=int, default=25)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/final_holdout/snapshot_draws.json"))
    parser.add_argument("--refresh-snapshot", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/final_holdout"))
    args = parser.parse_args()

    if (args.coarse + args.fine + args.local) > 300:
        raise ValueError("total search candidates must be <= 300")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_snapshot or not args.snapshot.exists():
        draws = fetch_live_draws(args.lookback_days)
        if len(draws) > args.max_eval_draws:
            draws = draws[-args.max_eval_draws :]
        save_snapshot(args.snapshot, draws)
        snapshot_source = "live_fetch_frozen"
    else:
        draws = load_snapshot(args.snapshot)
        snapshot_source = "frozen_snapshot"

    if len(draws) < 90:
        raise RuntimeError(f"insufficient draws for robust 3-way split: {len(draws)}")

    periods = [p for p, _ in draws]
    numbers = [d for _, d in draws]

    windows = split_windows(len(draws), 0.6, 0.2, 0.2, min_train_draws=20)
    final_blocks = split_final_blocks(*windows["final"])
    leak_free = leakage_check(windows, final_blocks)
    if not leak_free:
        raise RuntimeError("leakage check failed")

    search_start, search_end = windows["search"]
    val_start, val_end = windows["validation"]
    final_start, final_end = windows["final"]

    rng = random.Random(args.seed)
    search_log: list[dict[str, Any]] = []

    baseline = AppConfig(min_prediction_draws=10)
    _, baseline_search_b = evaluate_window(
        numbers, periods, search_start, search_end, baseline, args.seed, False
    )
    base_score = aggregate(baseline_search_b["model"])["top3_at_least_one_hit_rate"]

    candidates: dict[tuple[int, ...], dict[str, int]] = {}

    for _ in range(args.coarse):
        p = sample_params(rng)
        candidates[candidate_key(p)] = p

    coarse_rank: list[tuple[float, dict[str, int]]] = []
    for params in candidates.values():
        cfg = build_config(params)
        try:
            _, b = evaluate_window(numbers, periods, search_start, search_end, cfg, args.seed + 101, False)
            score = aggregate(b["model"])["top3_at_least_one_hit_rate"]
            exact3 = aggregate(b["model"])["exact_hit@3"]
        except PredictError:
            score = -1.0
            exact3 = -1.0
        search_log.append({"stage": "coarse", "score": score, "exact_hit@3": exact3, **params})
        coarse_rank.append((score, params))

    coarse_rank.sort(key=lambda x: x[0], reverse=True)
    seed_params = [p for _, p in coarse_rank[: min(8, len(coarse_rank))]]

    for _ in range(args.fine):
        base = seed_params[rng.randrange(len(seed_params))]
        p = refine_params(base, rng, scale=2)
        candidates[candidate_key(p)] = p

    for _ in range(args.local):
        base = seed_params[rng.randrange(len(seed_params))]
        p = refine_params(base, rng, scale=1)
        candidates[candidate_key(p)] = p

    validated: list[dict[str, Any]] = []
    for params in candidates.values():
        cfg = build_config(params)
        try:
            _, search_b = evaluate_window(numbers, periods, search_start, search_end, cfg, args.seed + 201, False)
            _, val_b = evaluate_window(numbers, periods, val_start, val_end, cfg, args.seed + 202, False)
            search_metrics = aggregate(search_b["model"])
            val_metrics = aggregate(val_b["model"])
            row = {
                "stage": "validate",
                "search_score": search_metrics["top3_at_least_one_hit_rate"],
                "validation_score": val_metrics["top3_at_least_one_hit_rate"],
                "validation_exact_hit@3": val_metrics["exact_hit@3"],
                **params,
            }
            validated.append(row)
            search_log.append(row)
        except PredictError:
            continue

    if not validated:
        raise RuntimeError("no valid candidate after search")

    validated.sort(
        key=lambda r: (
            r["validation_score"],
            r["validation_exact_hit@3"],
            r["search_score"],
        ),
        reverse=True,
    )
    best = validated[0]

    chosen_params = {
        k: int(best[k])
        for k in [
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
        ]
    }

    chosen_cfg = build_config(chosen_params)
    per_draw_rows, final_b = evaluate_window(
        numbers, periods, final_start, final_end, chosen_cfg, args.seed + 999, True
    )

    final_metrics = aggregate(final_b["model"])
    baseline_metrics = {k: aggregate(v) for k, v in final_b.items() if k != "model"}

    diffs = [
        final_b["model"][i]["top3_at_least_one_hit_rate"]
        - final_b["frequency"][i]["top3_at_least_one_hit_rate"]
        for i in range(len(final_b["model"]))
    ]
    p_value = paired_permutation_pvalue(diffs, random.Random(args.seed + 3000))
    ci_low, ci_high = bootstrap_ci(diffs, random.Random(args.seed + 3001))

    block_metrics: list[dict[str, Any]] = []
    for idx, (bs, be) in enumerate(final_blocks, start=1):
        _, b = evaluate_window(numbers, periods, bs, be, chosen_cfg, args.seed + 910 + idx, True)
        m = aggregate(b["model"])
        f = aggregate(b["frequency"])
        d = [
            b["model"][i]["top3_at_least_one_hit_rate"]
            - b["frequency"][i]["top3_at_least_one_hit_rate"]
            for i in range(len(b["model"]))
        ]
        pv = paired_permutation_pvalue(d, random.Random(args.seed + 3200 + idx))
        low, high = bootstrap_ci(d, random.Random(args.seed + 3300 + idx))
        block_metrics.append(
            {
                "block_id": idx,
                "issue_start": periods[bs],
                "issue_end": periods[be - 1],
                "draws": be - bs,
                "model_top3_at_least_one_hit_rate": m["top3_at_least_one_hit_rate"],
                "model_exact_hit@3": m["exact_hit@3"],
                "frequency_exact_hit@3": f["exact_hit@3"],
                "p_value_vs_frequency": pv,
                "bootstrap_ci_low_vs_frequency": low,
                "bootstrap_ci_high_vs_frequency": high,
            }
        )

    blocks_hit_ok = all(x["model_top3_at_least_one_hit_rate"] >= 0.80 for x in block_metrics)
    exact_ok = all(x["model_exact_hit@3"] >= x["frequency_exact_hit@3"] for x in block_metrics)
    stats_ok = all(
        x["p_value_vs_frequency"] < 0.05 and x["bootstrap_ci_low_vs_frequency"] > 0
        for x in block_metrics
    )
    passed = bool(blocks_hit_ok and exact_ok and stats_ok and leak_free)

    summary = {
        "snapshot_source": snapshot_source,
        "snapshot_fingerprint": snapshot_fingerprint(draws),
        "search_issue_range": [periods[search_start], periods[search_end - 1]],
        "validation_issue_range": [periods[val_start], periods[val_end - 1]],
        "final_holdout_issue_range": [periods[final_start], periods[final_end - 1]],
        "final_holdout_blocks": [
            {"block_id": i + 1, "issue_range": [periods[s], periods[e - 1]]}
            for i, (s, e) in enumerate(final_blocks)
        ],
        "total_draws_search": search_end - search_start,
        "total_draws_validation": val_end - val_start,
        "total_draws_final_holdout": final_end - final_start,
        "chosen_config": {
            "search_strategy": "coarse_fine_local_refine",
            "baseline_search_score": base_score,
            "params": chosen_params,
            "validation_score": best["validation_score"],
            "validation_exact_hit@3": best["validation_exact_hit@3"],
        },
        "final_metrics": final_metrics,
        "baseline_metrics": baseline_metrics,
        "block_metrics": block_metrics,
        "p_value_vs_frequency": p_value,
        "bootstrap_ci_vs_frequency": {"low": ci_low, "high": ci_high},
        "leakage_check_passed": leak_free,
        "passed": passed,
        "pass_reason": (
            "all_final_holdout_block_guardrails_passed"
            if passed
            else "failed_on_block_guardrails_or_exact_hit"
        ),
    }

    missing = REQUIRED_SUMMARY_KEYS - set(summary.keys())
    if missing:
        raise RuntimeError(f"summary missing keys: {sorted(missing)}")

    (args.out_dir / "summary_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "summary_report.md").write_text(
        "\n".join(
            [
                "# Final Holdout Validation",
                f"- snapshot_source: {summary['snapshot_source']}",
                f"- snapshot_fingerprint: {summary['snapshot_fingerprint']}",
                f"- search_issue_range: {summary['search_issue_range']}",
                f"- validation_issue_range: {summary['validation_issue_range']}",
                f"- final_holdout_issue_range: {summary['final_holdout_issue_range']}",
                f"- final_top3_at_least_one_hit_rate: {summary['final_metrics']['top3_at_least_one_hit_rate']:.6f}",
                f"- final_exact_hit@3: {summary['final_metrics']['exact_hit@3']:.6f}",
                f"- p_value_vs_frequency: {summary['p_value_vs_frequency']:.6f}",
                f"- bootstrap_ci_vs_frequency: {summary['bootstrap_ci_vs_frequency']}",
                f"- leakage_check_passed: {summary['leakage_check_passed']}",
                f"- passed: {summary['passed']}",
                f"- pass_reason: {summary['pass_reason']}",
            ]
        ),
        encoding="utf-8",
    )

    with (args.out_dir / "per_draw_report.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(per_draw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_draw_rows)

    per_block_rows: list[dict[str, Any]] = []
    for name, rows in final_b.items():
        per_block_rows.append({"segment": "final_holdout", "strategy": name, **aggregate(rows)})
    for row in block_metrics:
        per_block_rows.append({"segment": "final_holdout_block", "strategy": f"model_block_{row['block_id']}", **row})

    block_fields: list[str] = []
    for row in per_block_rows:
        for key in row.keys():
            if key not in block_fields:
                block_fields.append(key)

    with (args.out_dir / "per_block_report.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=block_fields)
        writer.writeheader()
        writer.writerows(per_block_rows)

    search_fields: list[str] = []
    for row in search_log:
        for key in row.keys():
            if key not in search_fields:
                search_fields.append(key)

    with (args.out_dir / "search_log.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=search_fields)
        writer.writeheader()
        writer.writerows(search_log)

    ablation = [
        {
            "variant": "baseline_default",
            "top3_at_least_one_hit_rate": base_score,
            "note": "default config on search window",
        },
        {
            "variant": "chosen_config_validation",
            "top3_at_least_one_hit_rate": best["validation_score"],
            "note": "selected by validation",
        },
        {
            "variant": "chosen_config_final",
            "top3_at_least_one_hit_rate": final_metrics["top3_at_least_one_hit_rate"],
            "note": "final untouched holdout",
        },
    ]
    with (args.out_dir / "ablation_report.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(ablation[0].keys()))
        writer.writeheader()
        writer.writerows(ablation)

    (args.out_dir / "best_config.json").write_text(
        json.dumps(summary["chosen_config"], indent=2), encoding="utf-8"
    )
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
                "metadata_only_knobs": [],
                "checked_entrypoint": "winwin_service.scoring.predict_top3",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (args.out_dir / "data_fingerprint.json").write_text(
        json.dumps(
            {
                "snapshot_path": str(args.snapshot),
                "snapshot_source": snapshot_source,
                "fingerprint": summary["snapshot_fingerprint"],
                "issue_range": [periods[0], periods[-1]],
                "total_draws": len(draws),
                "final_blocks": [
                    [periods[s], periods[e - 1]]
                    for s, e in final_blocks
                ],
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

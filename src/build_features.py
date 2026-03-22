from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils import safe_read_table, safe_write_table
from src.retrieval import RetrievalWeights, SimilarWindowRetriever, retrieval_features, retrieval_features_frame
from src.utils import DataContractError, DrawRecord, enforce_dir_file_sizes, ensure_numbers, log_progress, parse_date

WINDOWS = [20, 50, 100, 200, 500]


def _zone(n: int) -> int:
    return min((n - 1) // 20, 3)


def _consecutive_count(numbers: tuple[int, ...]) -> int:
    return sum(1 for a, b in zip(numbers, numbers[1:]) if b - a == 1)


def _gap_stats(history: list[DrawRecord], cand: int) -> tuple[float, float, float]:
    hit_idx = [i for i, row in enumerate(history) if cand in row.numbers]
    if len(hit_idx) < 2:
        current_gap = float(len(history) - hit_idx[-1]) if hit_idx else float(len(history) + 1)
        return current_gap, current_gap, 0.0
    gaps = [hit_idx[i] - hit_idx[i - 1] for i in range(1, len(hit_idx))]
    avg_gap = float(sum(gaps) / len(gaps))
    max_gap = float(max(gaps))
    current_gap = float(len(history) - hit_idx[-1])
    std = float((sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5)
    z = (current_gap - avg_gap) / std if std > 0 else 0.0
    return avg_gap, max_gap, z


def _history_profiles(history: list[DrawRecord], window: int) -> dict[str, float]:
    recent = history[-window:]
    zone_counts = [0, 0, 0, 0]
    odd_count = 0
    total_sum = 0
    total_span = 0
    total_consecutive = 0
    for row in recent:
        odd_count += sum(1 for n in row.numbers if n % 2 == 1)
        total_sum += sum(row.numbers)
        total_span += max(row.numbers) - min(row.numbers)
        total_consecutive += _consecutive_count(row.numbers)
        for n in row.numbers:
            zone_counts[_zone(n)] += 1
    denom = max(1, len(recent) * 20)
    zone_density = [z / denom for z in zone_counts]
    entropy = -sum(v * math.log(v) for v in zone_density if v > 0)
    return {
        "issue_zone_a_density": zone_density[0],
        "issue_zone_b_density": zone_density[1],
        "issue_zone_c_density": zone_density[2],
        "issue_zone_d_density": zone_density[3],
        "issue_zone_entropy": entropy,
        "issue_span": total_span / max(1, len(recent)),
        "issue_sum": total_sum / max(1, len(recent)),
        "issue_consecutive_count": total_consecutive / max(1, len(recent)),
        "issue_odd_even_balance": odd_count / denom,
        "issue_big_small_balance": (zone_density[2] + zone_density[3]) - (zone_density[0] + zone_density[1]),
        "issue_hot_zone": float(zone_density.index(max(zone_density))),
        "issue_compressed_zone": float(zone_density.index(min(zone_density))),
    }


def _regime_transition(history: list[DrawRecord]) -> float:
    if len(history) < 40:
        return 0.0
    old = _history_profiles(history[-40:-20], 20)["issue_zone_entropy"]
    new = _history_profiles(history[-20:], 20)["issue_zone_entropy"]
    return new - old


def _history_indicator_matrix(history: list[DrawRecord]) -> np.ndarray:
    mat = np.zeros((len(history), 80), dtype=np.float64)
    for i, row in enumerate(history):
        mat[i, np.asarray(row.numbers, dtype=np.int64) - 1] = 1.0
    return mat


def _rolling_hit_matrix(indicator: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for w in WINDOWS:
        span = indicator[-w:] if len(indicator) >= w else indicator
        out[w] = np.sum(span, axis=0)
    return out


def _candidate_gap_stats_matrix(indicator: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hist_len = indicator.shape[0]
    avg_gap = np.zeros(80, dtype=np.float64)
    max_gap = np.zeros(80, dtype=np.float64)
    gap_z = np.zeros(80, dtype=np.float64)
    current_gap = np.zeros(80, dtype=np.float64)
    for cand_idx in range(80):
        hits = np.flatnonzero(indicator[:, cand_idx] > 0)
        if len(hits) == 0:
            current = float(hist_len + 1)
            avg_gap[cand_idx] = current
            max_gap[cand_idx] = current
            gap_z[cand_idx] = 0.0
            current_gap[cand_idx] = current
            continue
        current = float(hist_len - hits[-1])
        current_gap[cand_idx] = current
        if len(hits) < 2:
            avg_gap[cand_idx] = current
            max_gap[cand_idx] = current
            gap_z[cand_idx] = 0.0
            continue
        gaps = np.diff(hits).astype(np.float64)
        avg = float(np.mean(gaps))
        mx = float(np.max(gaps))
        std = float(np.std(gaps))
        avg_gap[cand_idx] = avg
        max_gap[cand_idx] = mx
        gap_z[cand_idx] = (current - avg) / std if std > 0 else 0.0
    return current_gap, avg_gap, max_gap, gap_z


def _neighbor_hit_features_matrix(indicator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    recent50 = indicator[-50:] if len(indicator) >= 50 else indicator
    pm1 = np.zeros(80, dtype=np.float64)
    pm2 = np.zeros(80, dtype=np.float64)
    for idx in range(80):
        neighbors1 = []
        if idx - 1 >= 0:
            neighbors1.append(idx - 1)
        if idx + 1 < 80:
            neighbors1.append(idx + 1)
        neighbors2 = []
        if idx - 2 >= 0:
            neighbors2.append(idx - 2)
        if idx + 2 < 80:
            neighbors2.append(idx + 2)
        if neighbors1:
            pm1[idx] = float(np.sum(np.any(recent50[:, neighbors1] > 0, axis=1)))
        if neighbors2:
            pm2[idx] = float(np.sum(np.any(recent50[:, neighbors2] > 0, axis=1)))
    return pm1, pm2


def _recent_hot_cold_flags(history: list[DrawRecord], indicator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    recent_hot = {n for n, _ in Counter(num for r in history[-20:] for num in r.numbers).most_common(10)}
    freq_counter: Counter[int] = Counter()
    for row in history:
        freq_counter.update(row.numbers)
    recent_cold = {n for n, c in freq_counter.items() if c <= 2}
    hot_flag = np.zeros(80, dtype=np.float64)
    cold_flag = np.zeros(80, dtype=np.float64)
    for i in range(80):
        cand = i + 1
        hot_flag[i] = float(int((cand - 1) in recent_hot or (cand + 1) in recent_hot))
        cold_flag[i] = float(int((cand - 1) in recent_cold or (cand + 1) in recent_cold))
    return hot_flag, cold_flag


def build_history_runtime_cache(history: list[DrawRecord]) -> dict[str, object]:
    indicator = _history_indicator_matrix(history)
    rolling = _rolling_hit_matrix(indicator)
    total_hits = np.sum(indicator, axis=0)
    current_gap, avg_gap, max_gap, gap_z = _candidate_gap_stats_matrix(indicator)
    recent100 = indicator[-100:] if len(indicator) >= 100 else indicator
    decay_w = np.power(0.97, np.arange(len(recent100), dtype=np.float64))
    decay_100 = np.sum(recent100[::-1] * decay_w[:, None], axis=0)
    recent200 = indicator[-200:] if len(indicator) >= 200 else indicator
    ewma_w = np.power(0.95, np.arange(len(recent200), dtype=np.float64))
    ewma = np.sum(recent200[::-1] * ewma_w[:, None], axis=0)
    pm1, pm2 = _neighbor_hit_features_matrix(indicator)
    hot_flag, cold_flag = _recent_hot_cold_flags(history, indicator)
    carryover_prev = indicator[-1]
    carryover_last_k = np.sum(indicator[-10:] if len(indicator) >= 10 else indicator, axis=0)
    prof_10 = _history_profiles(history, min(10, len(history)))
    prof_20 = _history_profiles(history, min(20, len(history)))
    return {
        "indicator": indicator,
        "rolling": rolling,
        "total_hits": total_hits,
        "current_gap": current_gap,
        "avg_gap": avg_gap,
        "max_gap": max_gap,
        "gap_z": gap_z,
        "decay_100": decay_100,
        "ewma": ewma,
        "pm1": pm1,
        "pm2": pm2,
        "hot_flag": hot_flag,
        "cold_flag": cold_flag,
        "carryover_prev": carryover_prev,
        "carryover_last_k": carryover_last_k,
        "prof_10": prof_10,
        "prof_20": prof_20,
    }


def _build_candidate_rows_legacy_python(
    history: list[DrawRecord],
    issue: str,
    draw_date: str,
    label_numbers: set[int] | None,
    dynamic_n: int,
    matches: list,
    prof_10: dict[str, float],
    prof_20: dict[str, float],
    prof_n: dict[str, float],
    transition: float,
) -> list[dict[str, float | int | str]]:
    freq_counter: Counter[int] = Counter()
    for row in history:
        freq_counter.update(row.numbers)
    rows: list[dict[str, float | int | str]] = []
    for cand in range(1, 81):
        cand_hits = {w: sum(1 for r in history[-w:] if cand in r.numbers) for w in WINDOWS}
        hit_positions = [i for i, r in enumerate(history) if cand in r.numbers]
        current_gap = float(len(history) - hit_positions[-1]) if hit_positions else float(len(history) + 1)
        avg_gap, max_gap, gap_z = _gap_stats(history, cand)
        decay_100 = sum(0.97**i for i, r in enumerate(reversed(history[-100:])) if cand in r.numbers)
        ewma = sum((0.95**i) * (1.0 if cand in r.numbers else 0.0) for i, r in enumerate(reversed(history[-200:])))
        pm1 = sum(1 for r in history[-50:] if cand - 1 in r.numbers or cand + 1 in r.numbers)
        pm2 = sum(1 for r in history[-50:] if cand - 2 in r.numbers or cand + 2 in r.numbers)
        recent_hot = {n for n, _ in Counter(num for r in history[-20:] for num in r.numbers).most_common(10)}
        recent_cold = {n for n, c in freq_counter.items() if c <= 2}
        row: dict[str, float | int | str] = {
            "issue": issue,
            "draw_date": draw_date,
            "candidate_number": cand,
            "label": int(cand in label_numbers) if label_numbers is not None else -1,
            "day_issue_index": history[-1].day_issue_index,
            "day_total_seen_so_far": history[-1].day_issue_index,
            "normalized_day_progress": history[-1].day_issue_index / 999.0,
            "dynamic_context_n": dynamic_n,
            "cand_hits_last_20": float(cand_hits[20]),
            "cand_hits_last_50": float(cand_hits[50]),
            "cand_hits_last_100": float(cand_hits[100]),
            "cand_hits_last_200": float(cand_hits[200]),
            "cand_hits_last_500": float(cand_hits[500]),
            "cand_total_hits_all_time": float(freq_counter.get(cand, 0)),
            "cand_current_gap": current_gap,
            "cand_avg_gap": avg_gap,
            "cand_max_gap": max_gap,
            "cand_gap_zscore": gap_z,
            "cand_recent_hit_decay": decay_100,
            "cand_ewma_freq": ewma,
            "cand_pm1_neighbor_hits": float(pm1),
            "cand_pm2_neighbor_hits": float(pm2),
            "cand_pm1_neighbor_decay": pm1 / 50.0,
            "cand_pm2_neighbor_decay": pm2 / 50.0,
            "cand_neighbor_balance": float(pm1 - pm2),
            "cand_is_adjacent_to_recent_hot": float(int((cand - 1) in recent_hot or (cand + 1) in recent_hot)),
            "cand_is_adjacent_to_recent_cold": float(int((cand - 1) in recent_cold or (cand + 1) in recent_cold)),
            "cand_carryover_from_prev_draw": float(int(cand in history[-1].numbers)),
            "cand_carryover_count_last_k": float(sum(1 for r in history[-10:] if cand in r.numbers)),
            "cand_recent_reactivation_score": decay_100 / max(1.0, current_gap),
            "cand_handoff_score": (pm1 + pm2) / 100.0,
            "cand_rebound_score": max(0.0, gap_z),
            "issue_transition_regime": transition,
            "current_day_recent_10_profile": json.dumps(prof_10, ensure_ascii=False, sort_keys=True),
            "current_day_recent_20_profile": json.dumps(prof_20, ensure_ascii=False, sort_keys=True),
            "current_day_recent_n_profile": json.dumps(prof_n, ensure_ascii=False, sort_keys=True),
        }
        row.update(prof_n)
        row.update(retrieval_features(matches, cand, context_n=dynamic_n))
        rows.append(row)
    return rows


def resolve_dynamic_context(history: list[DrawRecord], min_dynamic_n: int, max_dynamic_n: int) -> list[DrawRecord]:
    if not history:
        raise DataContractError("history is empty")
    day = history[-1].draw_date
    same_day_rows = [r for r in history if r.draw_date == day]
    context = same_day_rows if same_day_rows else history
    if len(context) > max_dynamic_n:
        context = context[-max_dynamic_n:]
    if len(context) < min_dynamic_n:
        raise DataContractError(f"dynamic context too short: N={len(context)} < min_dynamic_n={min_dynamic_n}")
    return context


def build_candidate_rows(
    history: list[DrawRecord],
    issue: str,
    draw_date: str,
    label_numbers: set[int] | None,
    min_dynamic_n: int,
    max_dynamic_n: int,
    top_k: int,
    retrieval_weights: dict[str, float] | None = None,
    prefer_same_day_progress: bool = True,
    progress_logging: bool = False,
    runtime_cache: dict[str, object] | None = None,
) -> tuple[list[dict[str, float | int | str]], list]:
    if len(history) < min_dynamic_n + 1:
        raise DataContractError("insufficient history for candidate feature generation")

    context = resolve_dynamic_context(history, min_dynamic_n=min_dynamic_n, max_dynamic_n=max_dynamic_n)
    dynamic_n = len(context)
    if progress_logging:
        log_progress(1, 5, "dynamic_n 已解析", f"dynamic_n={dynamic_n}")
    retriever = SimilarWindowRetriever(
        top_k=top_k,
        weights=RetrievalWeights.from_mapping(retrieval_weights),
        require_same_length_window=True,
        prefer_same_day_progress=prefer_same_day_progress,
    )
    if progress_logging:
        log_progress(2, 5, "開始建立 prediction candidate rows", f"issue={issue}")
    matches = retriever.query(history=history, target_window=context, day_issue_index=context[-1].day_issue_index)
    if progress_logging:
        log_progress(3, 5, "retrieval matches 完成", f"match_count={len(matches)}")

    cache = runtime_cache or build_history_runtime_cache(history)
    prof_10 = dict(cache["prof_10"])  # type: ignore[arg-type]
    prof_20 = dict(cache["prof_20"])  # type: ignore[arg-type]
    prof_n = _history_profiles(history, min(len(history), max(20, dynamic_n)))
    transition = _regime_transition(history)
    rolling = cache["rolling"]  # type: ignore[assignment]
    total_hits = cache["total_hits"]  # type: ignore[assignment]
    current_gap = cache["current_gap"]  # type: ignore[assignment]
    avg_gap = cache["avg_gap"]  # type: ignore[assignment]
    max_gap = cache["max_gap"]  # type: ignore[assignment]
    gap_z = cache["gap_z"]  # type: ignore[assignment]
    decay_100 = cache["decay_100"]  # type: ignore[assignment]
    ewma = cache["ewma"]  # type: ignore[assignment]
    pm1 = cache["pm1"]  # type: ignore[assignment]
    pm2 = cache["pm2"]  # type: ignore[assignment]
    hot_flag = cache["hot_flag"]  # type: ignore[assignment]
    cold_flag = cache["cold_flag"]  # type: ignore[assignment]
    carryover_prev = cache["carryover_prev"]  # type: ignore[assignment]
    carryover_last_k = cache["carryover_last_k"]  # type: ignore[assignment]
    retrieval_frame = retrieval_features_frame(matches, context_n=dynamic_n)

    profile10 = json.dumps(prof_10, ensure_ascii=False, sort_keys=True)
    profile20 = json.dumps(prof_20, ensure_ascii=False, sort_keys=True)
    profilen = json.dumps(prof_n, ensure_ascii=False, sort_keys=True)
    rows: list[dict[str, float | int | str]] = []
    for cand in range(1, 81):
        idx = cand - 1
        row: dict[str, float | int | str] = {
            "issue": issue,
            "draw_date": draw_date,
            "candidate_number": cand,
            "label": int(cand in label_numbers) if label_numbers is not None else -1,
            "day_issue_index": history[-1].day_issue_index,
            "day_total_seen_so_far": history[-1].day_issue_index,
            "normalized_day_progress": history[-1].day_issue_index / 999.0,
            "dynamic_context_n": dynamic_n,
            "cand_hits_last_20": float(rolling[20][idx]),
            "cand_hits_last_50": float(rolling[50][idx]),
            "cand_hits_last_100": float(rolling[100][idx]),
            "cand_hits_last_200": float(rolling[200][idx]),
            "cand_hits_last_500": float(rolling[500][idx]),
            "cand_total_hits_all_time": float(total_hits[idx]),
            "cand_current_gap": float(current_gap[idx]),
            "cand_avg_gap": float(avg_gap[idx]),
            "cand_max_gap": float(max_gap[idx]),
            "cand_gap_zscore": float(gap_z[idx]),
            "cand_recent_hit_decay": float(decay_100[idx]),
            "cand_ewma_freq": float(ewma[idx]),
            "cand_pm1_neighbor_hits": float(pm1[idx]),
            "cand_pm2_neighbor_hits": float(pm2[idx]),
            "cand_pm1_neighbor_decay": float(pm1[idx] / 50.0),
            "cand_pm2_neighbor_decay": float(pm2[idx] / 50.0),
            "cand_neighbor_balance": float(pm1[idx] - pm2[idx]),
            "cand_is_adjacent_to_recent_hot": float(hot_flag[idx]),
            "cand_is_adjacent_to_recent_cold": float(cold_flag[idx]),
            "cand_carryover_from_prev_draw": float(carryover_prev[idx]),
            "cand_carryover_count_last_k": float(carryover_last_k[idx]),
            "cand_recent_reactivation_score": float(decay_100[idx] / max(1.0, current_gap[idx])),
            "cand_handoff_score": float((pm1[idx] + pm2[idx]) / 100.0),
            "cand_rebound_score": float(max(0.0, gap_z[idx])),
            "issue_transition_regime": transition,
            "current_day_recent_10_profile": profile10,
            "current_day_recent_20_profile": profile20,
            "current_day_recent_n_profile": profilen,
        }
        row.update(prof_n)
        row.update(retrieval_frame[cand])
        rows.append(row)
    if progress_logging:
        log_progress(4, 5, "prediction candidate rows 完成", f"rows={len(rows)}")
        log_progress(5, 5, "feature contract passed", f"dynamic_n={dynamic_n}")
    return rows, matches


def build_feature_rows(
    records: list[DrawRecord],
    min_history: int,
    min_dynamic_n: int = 20,
    max_dynamic_n: int = 999,
    top_k: int = 50,
    retrieval_weights: dict[str, float] | None = None,
    prefer_same_day_progress: bool = True,
    retrieval_window: int | None = None,
) -> list[dict[str, float | int | str]]:
    if len(records) <= min_history:
        raise DataContractError("insufficient records for feature generation")
    if retrieval_window is not None:
        max_dynamic_n = min(max_dynamic_n, int(retrieval_window))

    rows: list[dict[str, float | int | str]] = []
    total_steps = max(1, len(records) - min_history)
    last_logged = -1
    for idx in range(min_history, len(records)):
        target_next = records[idx]
        history = records[:idx]
        context_issue = history[-1]
        progress = int((idx - min_history + 1) / total_steps * 100)
        if progress // 10 != last_logged:
            last_logged = progress // 10
            log_progress(
                idx - min_history + 1,
                total_steps,
                "建立候選特徵",
                f"issue_t={context_issue.issue} -> issue_t+1={target_next.issue}",
            )
        try:
            built, _ = build_candidate_rows(
                history=history,
                issue=context_issue.issue,
                draw_date=context_issue.draw_date.isoformat(),
                label_numbers=set(target_next.numbers),
                min_dynamic_n=min_dynamic_n,
                max_dynamic_n=max_dynamic_n,
                top_k=top_k,
                retrieval_weights=retrieval_weights,
                prefer_same_day_progress=prefer_same_day_progress,
            )
            rows.extend(built)
        except DataContractError:
            continue
    return rows


def write_feature_store(path: Path, rows: list[dict[str, float | int | str]]) -> Path:
    if not rows:
        raise DataContractError("feature row output is empty")
    frame = pd.DataFrame(rows)
    return safe_write_table(
        frame,
        path,
        max_file_mb=95,
        preferred_format="parquet",
        producer_script="src.build_features",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/history_processed.csv")
    parser.add_argument("--output", default="data/feature_store/ranking_features.csv")
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--min-dynamic-n", type=int, default=20)
    parser.add_argument("--max-dynamic-n", type=int, default=999)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    log_progress(1, 3, "讀取 processed 歷史", f"輸入={args.input}")
    history_df = safe_read_table(Path(args.input))
    required = {"issue", "draw_date", "numbers", "day_issue_index"}
    missing = required - set(history_df.columns)
    if missing:
        raise DataContractError(f"processed history missing columns: {sorted(missing)}")
    records = [
        DrawRecord(
            issue=str(row.issue),
            draw_date=parse_date(str(row.draw_date)),
            numbers=ensure_numbers(json.loads(str(row.numbers))),
            day_issue_index=int(row.day_issue_index),
        )
        for row in history_df.itertuples(index=False)
    ]
    log_progress(2, 3, "開始建立 ranking features", f"歷史筆數={len(records)}")
    rows = build_feature_rows(records, args.min_history, args.min_dynamic_n, args.max_dynamic_n, args.top_k)
    out = write_feature_store(Path(args.output), rows)
    enforce_dir_file_sizes([Path("data/feature_store"), Path("reports"), Path("models")])
    log_progress(3, 3, "feature store 輸出完成", f"輸出={out}")


if __name__ == "__main__":
    main()

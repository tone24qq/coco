from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from src.io_utils import safe_read_table, safe_write_table
from src.retrieval import RetrievalWeights, SimilarWindowRetriever, retrieval_features
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
) -> tuple[list[dict[str, float | int | str]], list]:
    if len(history) < min_dynamic_n + 1:
        raise DataContractError("insufficient history for candidate feature generation")

    context = resolve_dynamic_context(history, min_dynamic_n=min_dynamic_n, max_dynamic_n=max_dynamic_n)
    dynamic_n = len(context)
    retriever = SimilarWindowRetriever(
        top_k=top_k,
        weights=RetrievalWeights.from_mapping(retrieval_weights),
        require_same_length_window=True,
        prefer_same_day_progress=prefer_same_day_progress,
    )
    matches = retriever.query(history=history, target_window=context, day_issue_index=context[-1].day_issue_index)

    prof_10 = _history_profiles(history, min(10, len(history)))
    prof_20 = _history_profiles(history, min(20, len(history)))
    prof_n = _history_profiles(history, min(len(history), max(20, dynamic_n)))
    transition = _regime_transition(history)

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

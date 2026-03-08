from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_STORE_DIR = ROOT / "data" / "feature_store"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "configs"

RAW_FILES = [
    "賓果賓果_2023.csv",
    "賓果賓果_2024.csv",
    "賓果賓果_2025.csv",
    "賓果賓果_2026.csv",
]
ZONE_NAMES = ["A", "B", "C", "D"]


@dataclass
class TrainConfig:
    feature_min_history: int = 20
    backtest_splits: int = 5
    lgbm_params: Dict[str, float] | None = None


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for p in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        FEATURE_STORE_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def parse_numbers(row: pd.Series) -> List[int]:
    cols = [f"獎號{i}" for i in range(1, 21)]
    if not all(c in row.index for c in cols):
        cols = [f"n{i}" for i in range(1, 21)]
    nums = [int(row[c]) for c in cols]
    nums = sorted(nums)
    if len(nums) != 20:
        raise ValueError("Each draw must have 20 numbers")
    return nums


def zone_of(num: int) -> str:
    if num <= 20:
        return "A"
    if num <= 40:
        return "B"
    if num <= 60:
        return "C"
    return "D"


def classify_board(zone_counts: Dict[str, int]) -> str:
    values = [zone_counts[z] for z in ZONE_NAMES]
    if max(values) - min(values) <= 1:
        return "balanced"
    s = sorted(values, reverse=True)
    if s[0] >= 9:
        return "single_zone_burst"
    if s[0] >= 7 and s[1] >= 6 and s[2] <= 4:
        return "double_zone_shake"
    return "extreme_burst"


def _tail_features(numbers: Sequence[int]) -> Dict[str, float]:
    tails = [n % 10 for n in numbers]
    values, counts = np.unique(tails, return_counts=True)
    counter = {int(v): int(c) for v, c in zip(values, counts)}
    data = {f"tail_{i}_cnt": counter.get(i, 0) for i in range(10)}
    data["tail_unique_cnt"] = float(len(values))
    data["tail_max_cnt"] = float(max(counter.values()))
    data["same_tail_pair_cnt"] = float(sum(c * (c - 1) / 2 for c in counter.values()))
    return data


def _gap_features(numbers: Sequence[int]) -> Dict[str, float]:
    arr = np.array(sorted(numbers))
    gaps = np.diff(arr)
    consecutive_pairs = int((gaps == 1).sum())
    run = 1
    max_run = 1
    for g in gaps:
        if g == 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return {
        "consecutive_pairs": float(consecutive_pairs),
        "max_consecutive_run": float(max_run),
        "run_len_ge_3": float(max_run >= 3),
        "run_len_ge_4": float(max_run >= 4),
        "gap_mean": float(gaps.mean()),
        "gap_std": float(gaps.std()),
        "gap_min": float(gaps.min()),
        "gap_max": float(gaps.max()),
        "gap_lt_3_cnt": float((gaps < 3).sum()),
        "gap_ge_8_cnt": float((gaps >= 8).sum()),
    }


def _zone_counts(numbers: Sequence[int]) -> Dict[str, int]:
    return {
        "A": int(sum(1 for n in numbers if n <= 20)),
        "B": int(sum(1 for n in numbers if 21 <= n <= 40)),
        "C": int(sum(1 for n in numbers if 41 <= n <= 60)),
        "D": int(sum(1 for n in numbers if 61 <= n <= 80)),
    }


def _recent_frequency(history: Sequence[Sequence[int]], window: int) -> Dict[int, int]:
    draws = history[-window:]
    freq = {n: 0 for n in range(1, 81)}
    for d in draws:
        for n in d:
            freq[n] += 1
    return freq


def build_issue_features(df: pd.DataFrame, min_history: int = 20) -> pd.DataFrame:
    rows: list[dict] = []
    draws = [json.loads(v) if isinstance(v, str) else v for v in df["numbers"]]
    board_history: list[dict] = []

    for i in range(min_history, len(df) - 1):
        nums = sorted(draws[i])
        hist = draws[: i + 1]
        zc = _zone_counts(nums)
        arr = np.array(nums)
        big_cnt = int((arr > 40).sum())
        odd_cnt = int((arr % 2 == 1).sum())

        feat: dict[str, float | int | str] = {
            "issue": int(df.iloc[i]["issue"]),
            "draw_date": str(df.iloc[i]["draw_date"]),
            "target_issue": int(df.iloc[i + 1]["issue"]),
            "target_numbers": json.dumps(sorted(draws[i + 1]), ensure_ascii=False),
            "sum_all": float(arr.sum()),
            "mean_all": float(arr.mean()),
            "std_all": float(arr.std()),
            "min_num": float(arr.min()),
            "max_num": float(arr.max()),
            "span": float(arr.max() - arr.min()),
            "median": float(np.median(arr)),
            "zone_A_cnt": float(zc["A"]),
            "zone_B_cnt": float(zc["B"]),
            "zone_C_cnt": float(zc["C"]),
            "zone_D_cnt": float(zc["D"]),
            "zone_range": float(max(zc.values()) - min(zc.values())),
            "dominant_zone": float(np.argmax([zc[z] for z in ZONE_NAMES])),
            "is_balanced_5_5_5_5": float(all(v == 5 for v in zc.values())),
            "is_single_zone_burst": float(max(zc.values()) >= 9),
            "is_double_zone_shake": float(
                sorted(zc.values(), reverse=True)[0] >= 7
                and sorted(zc.values(), reverse=True)[1] >= 6
            ),
            "is_extreme_burst": float(max(zc.values()) >= 10),
            "small_cnt": float(20 - big_cnt),
            "big_cnt": float(big_cnt),
            "big_minus_small": float(big_cnt - (20 - big_cnt)),
            "odd_cnt": float(odd_cnt),
            "even_cnt": float(20 - odd_cnt),
            "odd_minus_even": float(odd_cnt - (20 - odd_cnt)),
            "big_odd_cnt": float(sum(1 for n in nums if n > 40 and n % 2 == 1)),
            "big_even_cnt": float(sum(1 for n in nums if n > 40 and n % 2 == 0)),
            "small_odd_cnt": float(sum(1 for n in nums if n <= 40 and n % 2 == 1)),
            "small_even_cnt": float(sum(1 for n in nums if n <= 40 and n % 2 == 0)),
        }
        feat.update(_gap_features(nums))
        feat.update(_tail_features(nums))

        for w in [3, 5, 10, 20]:
            freq = _recent_frequency(hist, w)
            feat[f"recent_freq_{w}"] = float(np.mean([freq[n] for n in nums]))
        freq20 = _recent_frequency(hist, 20)
        hot_scores = np.array([freq20[n] for n in nums])
        feat["hot_score_mean"] = float(hot_scores.mean())
        feat["hot_score_max"] = float(hot_scores.max())
        feat["cold_score_cnt"] = float(
            (hot_scores <= np.percentile(list(freq20.values()), 30)).sum()
        )

        prev = board_history[-1] if board_history else None
        prev3 = board_history[-3] if len(board_history) >= 3 else None
        feat["delta_sum_1"] = float(
            feat["sum_all"] - (prev["sum_all"] if prev else feat["sum_all"])
        )
        feat["delta_sum_3"] = float(
            feat["sum_all"] - (prev3["sum_all"] if prev3 else feat["sum_all"])
        )
        feat["delta_big_small_1"] = float(
            feat["big_minus_small"]
            - (prev["big_minus_small"] if prev else feat["big_minus_small"])
        )
        feat["delta_odd_even_1"] = float(
            feat["odd_minus_even"]
            - (prev["odd_minus_even"] if prev else feat["odd_minus_even"])
        )
        for z in ZONE_NAMES:
            current = feat[f"zone_{z}_cnt"]
            previous = prev[f"zone_{z}_cnt"] if prev else current
            feat[f"delta_zone_{z}_1"] = float(current - previous)
            feat[f"abs_delta_zone_{z}_1"] = float(abs(current - previous))
        feat["delta_gap_mean_1"] = float(
            feat["gap_mean"] - (prev["gap_mean"] if prev else feat["gap_mean"])
        )
        feat["delta_consecutive_pairs_1"] = float(
            feat["consecutive_pairs"]
            - (prev["consecutive_pairs"] if prev else feat["consecutive_pairs"])
        )

        history_df = pd.DataFrame(board_history)
        for w in [3, 5]:
            if len(history_df) >= w:
                feat[f"roll{w}_sum_mean"] = float(history_df["sum_all"].tail(w).mean())
                feat[f"roll{w}_sum_std"] = float(
                    history_df["sum_all"].tail(w).std(ddof=0)
                )
            else:
                feat[f"roll{w}_sum_mean"] = float(feat["sum_all"])
                feat[f"roll{w}_sum_std"] = 0.0
        for z in ZONE_NAMES:
            if len(history_df) >= 5:
                feat[f"roll5_zone_{z}_mean"] = float(
                    history_df[f"zone_{z}_cnt"].tail(5).mean()
                )
            else:
                feat[f"roll5_zone_{z}_mean"] = float(feat[f"zone_{z}_cnt"])
            if len(history_df) >= 10:
                feat[f"roll10_zone_{z}_std"] = float(
                    history_df[f"zone_{z}_cnt"].tail(10).std(ddof=0)
                )
            else:
                feat[f"roll10_zone_{z}_std"] = 0.0
        feat["sum_vs_roll3_mean"] = float(feat["sum_all"] - feat["roll3_sum_mean"])
        for z in ZONE_NAMES:
            feat[f"zone_{z}_vs_roll5_mean"] = float(
                feat[f"zone_{z}_cnt"] - feat[f"roll5_zone_{z}_mean"]
            )
        feat["big_small_vs_roll5_mean"] = float(
            feat["big_minus_small"]
            - (
                history_df["big_minus_small"].tail(5).mean()
                if len(history_df) >= 1
                else feat["big_minus_small"]
            )
        )
        feat["odd_even_vs_roll5_mean"] = float(
            feat["odd_minus_even"]
            - (
                history_df["odd_minus_even"].tail(5).mean()
                if len(history_df) >= 1
                else feat["odd_minus_even"]
            )
        )

        if len(board_history) >= 6:
            candidates = []
            cur_vec = np.array(
                [
                    feat["zone_A_cnt"],
                    feat["zone_B_cnt"],
                    feat["zone_C_cnt"],
                    feat["zone_D_cnt"],
                    feat["big_minus_small"],
                    feat["odd_minus_even"],
                    feat["sum_all"],
                ]
            )
            for idx in range(len(board_history) - 1):
                past = board_history[idx]
                pvec = np.array(
                    [
                        past["zone_A_cnt"],
                        past["zone_B_cnt"],
                        past["zone_C_cnt"],
                        past["zone_D_cnt"],
                        past["big_minus_small"],
                        past["odd_minus_even"],
                        past["sum_all"],
                    ]
                )
                dist = np.linalg.norm(cur_vec - pvec)
                score = 1.0 / (1.0 + dist)
                candidates.append((score, idx))
            candidates.sort(reverse=True, key=lambda x: x[0])
            top5 = candidates[:5]
            feat["sim_top1_score"] = float(top5[0][0])
            feat["sim_top5_mean_score"] = float(np.mean([s for s, _ in top5]))
            following = [board_history[idx + 1] for _, idx in top5]
            for z in ZONE_NAMES:
                feat[f"sim_following_zone_{z}_mean"] = float(
                    np.mean([x[f"zone_{z}_cnt"] for x in following])
                )
            feat["sim_following_big_small_mean"] = float(
                np.mean([x["big_minus_small"] for x in following])
            )
            feat["sim_following_odd_even_mean"] = float(
                np.mean([x["odd_minus_even"] for x in following])
            )
        else:
            feat["sim_top1_score"] = 0.0
            feat["sim_top5_mean_score"] = 0.0
            for z in ZONE_NAMES:
                feat[f"sim_following_zone_{z}_mean"] = float(feat[f"zone_{z}_cnt"])
            feat["sim_following_big_small_mean"] = float(feat["big_minus_small"])
            feat["sim_following_odd_even_mean"] = float(feat["odd_minus_even"])

        rows.append(feat)
        board_history.append(
            {
                k: feat[k]
                for k in feat
                if k not in {"issue", "draw_date", "target_issue", "target_numbers"}
            }
        )

    return pd.DataFrame(rows)


def issue_feature_columns(df: pd.DataFrame) -> List[str]:
    skip = {"issue", "draw_date", "target_issue", "target_numbers"}
    return [c for c in df.columns if c not in skip]


def build_candidate_matrix(
    issue_row: pd.Series, feature_columns: Sequence[str]
) -> pd.DataFrame:
    base = issue_row.to_dict()
    rows = []
    for num in range(1, 81):
        row = {
            k: base.get(k, 0.0)
            for k in feature_columns
            if k not in {"num", "num_norm", "num_zone", "num_is_odd", "num_is_big"}
        }
        row["num"] = float(num)
        row["num_norm"] = float(num / 80)
        row["num_zone"] = float((num - 1) // 20)
        row["num_is_odd"] = float(num % 2 == 1)
        row["num_is_big"] = float(num > 40)
        rows.append(row)
    return pd.DataFrame(rows)[list(feature_columns)]


def compact_10_from_top20(top20: Sequence[int]) -> List[int]:
    buckets = {z: [] for z in ZONE_NAMES}
    for n in top20:
        buckets[zone_of(n)].append(n)
    compact = []
    while len(compact) < 10 and any(buckets.values()):
        for z in ZONE_NAMES:
            if buckets[z] and len(compact) < 10:
                compact.append(buckets[z].pop(0))
    return compact


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_processed() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED_DIR / "bingo_draws.csv")

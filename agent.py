from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
CSV_FILES = [
    "賓果賓果_2023.csv",
    "賓果賓果_2024.csv",
    "賓果賓果_2025.csv",
    "賓果賓果_2026.csv",
]
DEFAULT_SEED = 42
PREDICT_REQUIRED_MESSAGE = "請先提供最新 10–50 期資料（每期20顆），才可進行下一期預測。"
HISTORY_MIN_THRESHOLD = 200
DEFAULT_LAST_DRAW_PENALTY = 0.35
RELAXED_LAST_DRAW_PENALTY = 0.20
DEFAULT_LAST_DRAW_MAX_IN_TOPK = 4
PRIME_SET = {
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
}
logger = logging.getLogger(__name__)

DEFAULT_SEQUENCE_SIMILARITY_WEIGHTS = {
    "number_overlap_score": 0.20,
    "zone_structure_score": 0.25,
    "consecutive_pattern_score": 0.15,
    "trend_pattern_score": 0.20,
    "dynamic_board_type_score": 0.20,
}

DEFAULT_SEQUENCE_SCORE_WEIGHTS = {
    "similar_following_frequency": 0.40,
    "similarity_weighted_frequency": 0.40,
    "current_pattern_adjustment": 0.20,
}

FEATURE_VERSION = "v2.0-standardized-5board"
SIMILARITY_VERSION = "v2.0-two-stage-prefilter"
ADJUSTMENT_VERSION = "v2.0-event-effect-calibrated"
ZONE_LABELS = ["A", "B", "C", "D"]
WEIGHT_TABLE = {
    "小盤": (0.30, 0.40, 0.30),
    "平均盤": (0.20, 0.50, 0.30),
    "大盤": (0.15, 0.55, 0.30),
}


# 歷史主導 baseline v1
@dataclass(frozen=True)
class ScoreWeights:
    recent_momentum: float = 0
    zone_distribution: float = 0.22
    pattern_similarity: float = 0.24
    hot_frequency: float = 0.12
    big_mid_small: float = 0.07
    consecutive_pattern: float = 0.05
    cluster_pattern: float = 0.12
    tail_concentration: float = 0.03
    gap_skip_pattern: float = 0.03
    sum_range: float = 0.05
    odd_even_balance: float = 0.04
    delta_pattern: float = 0.03
    skip_heat: float = 0.03
    prime_boost: float = 0.02
    compression_boost: float = 0.04

    def as_dict(self) -> Dict[str, float]:
        weights = asdict(self)
        total = float(sum(weights.values()))
        if total <= 0:
            return weights
        return {k: float(v / total) for k, v in weights.items()}


class RecentDraw(BaseModel):
    issue: int
    numbers: List[int] = Field(..., min_length=20, max_length=20)


class PredictRequest(BaseModel):
    recent: List[RecentDraw] = Field(..., min_length=10, max_length=50)
    top_k: int = Field(default=20, ge=1, le=20)


class PredictTop3Request(PredictRequest):
    window: Optional[int] = Field(default=None, ge=10, le=50)
    alpha: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    lambda_: Optional[float] = Field(default=None, alias="lambda", ge=0.0)
    candidate_pool_size: Optional[int] = Field(default=None, ge=8, le=30)


class BacktestRequest(BaseModel):
    windows: List[int] = Field(default_factory=lambda: [50, 100, 200])
    alphas: List[float] = Field(default_factory=lambda: [0.7, 0.8, 0.9, 0.95])
    lambdas: List[float] = Field(default_factory=lambda: [0.3, 0.8, 1.5, 2.5])
    recent_n: int = Field(default=200, ge=20)
    candidate_pool_size: int = Field(default=18, ge=8, le=30)
    random_runs: int = Field(default=500, ge=100, le=5000)
    max_steps: Optional[int] = Field(default=None, ge=1)
    output_dir: str = Field(default="artifacts")


class WalkForwardRequest(BaseModel):
    train_window: int = Field(default=200, ge=50)
    max_steps: Optional[int] = Field(default=200, ge=1)


class SequenceSimilarityPredictRequest(PredictRequest):
    input_window_size: int = Field(default=10, ge=5, le=50)
    min_match_count: int = Field(default=10, ge=3, le=100)
    top_k: int = Field(default=15, ge=3, le=100)
    output_top_n: int = Field(default=10, ge=3, le=80)
    min_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    similarity_weights: Optional[Dict[str, float]] = None
    score_weights: Optional[Dict[str, float]] = None


class SequenceSimilarityBacktestRequest(BaseModel):
    input_window_size: int = Field(default=10, ge=5, le=50)
    min_match_count: int = Field(default=10, ge=3, le=100)
    top_k: int = Field(default=15, ge=3, le=100)
    output_top_n: int = Field(default=10, ge=3, le=80)
    min_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    similarity_weights: Optional[Dict[str, float]] = None
    score_weights: Optional[Dict[str, float]] = None
    max_steps: Optional[int] = Field(default=200, ge=1)


class BingoAnalyzer:
    def __init__(
        self,
        csv_path: Path | str | None = None,
        csv_files: Sequence[Path | str] | None = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.csv_files = [Path(path) for path in csv_files] if csv_files else None
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.df = self._load_and_prepare_data()
        self.draw_numbers: List[List[int]] = self._extract_draw_numbers(self.df)
        self.recent_30_draws = self.draw_numbers[-30:]
        self.recent_30_tail_stats = self._tail_digit_stats(self.recent_30_draws)
        self.recent_30_sums = [sum(draw) for draw in self.recent_30_draws]
        self.matrix = self._build_matrix(self.draw_numbers)
        self.issue_to_index = {
            int(issue): idx for idx, issue in enumerate(self.df["issue"].tolist())
        }
        self.history_verification = self._verify_history_usage()

    def _verify_history_usage(self, recent_window: int = 50) -> Dict[str, object]:
        loaded = len(self.draw_numbers)
        recent_used = min(recent_window, loaded)
        baseline = max(0, loaded - recent_used)
        warnings: List[str] = []
        if loaded < HISTORY_MIN_THRESHOLD:
            warnings.append(
                f"history rows {loaded} below threshold {HISTORY_MIN_THRESHOLD}"
            )
        logger.info(
            "Loaded draws: %s | Recent window: %s | History baseline: %s",
            loaded,
            recent_used,
            baseline,
        )
        return {
            "loaded_draws": loaded,
            "recent_window": recent_used,
            "history_baseline": baseline,
            "warnings": warnings,
        }

    @staticmethod
    def _resolve_csv_path(path: Path) -> Path:
        if path.exists():
            return path
        data_candidate = BASE_DIR / "data" / path.name
        if data_candidate.exists():
            return data_candidate
        base_candidate = BASE_DIR / path.name
        if base_candidate.exists():
            return base_candidate
        raise FileNotFoundError(f"CSV file not found: {path}")

    def _load_and_prepare_data(self) -> pd.DataFrame:
        if self.csv_path is not None:
            target_files = [self._resolve_csv_path(self.csv_path)]
        elif self.csv_files is not None:
            target_files = [self._resolve_csv_path(path) for path in self.csv_files]
        else:
            target_files = [self._resolve_csv_path(Path(name)) for name in CSV_FILES]

        dfs = [pd.read_csv(path) for path in target_files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.copy()
        if "issue" not in df.columns and "期別" in df.columns:
            df = df.rename(columns={"期別": "issue"})
        if "issue" not in df.columns:
            raise ValueError("CSV must include issue/期別 column")
        df["issue"] = pd.to_numeric(df["issue"], errors="coerce")
        df = df.dropna(subset=["issue"]).sort_values("issue").reset_index(drop=True)
        df["issue"] = df["issue"].astype(int)
        return df

    def _extract_draw_numbers(self, df: pd.DataFrame) -> List[List[int]]:
        zhong_cols = [c for c in df.columns if str(c).startswith("獎號")]
        if len(zhong_cols) >= 20:
            ordered = sorted(zhong_cols, key=lambda x: int(str(x).replace("獎號", "")))
            draws = (
                df[ordered[:20]]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
                .values.tolist()
            )
            return [self._validate_draw(row) for row in draws]

        ncols = [
            c for c in df.columns if str(c).startswith("n") and str(c)[1:].isdigit()
        ]
        if len(ncols) >= 20:
            ordered = sorted(ncols, key=lambda x: int(str(x)[1:]))
            draws = (
                df[ordered[:20]]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
                .values.tolist()
            )
            return [self._validate_draw(row) for row in draws]

        num_cols = [
            c for c in df.columns if str(c).isdigit() and 1 <= int(str(c)) <= 80
        ]
        if len(num_cols) == 80:
            ordered = sorted(num_cols, key=lambda x: int(x))
            binary_rows = (
                df[ordered]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
                .values
            )
            draws: List[List[int]] = []
            for row in binary_rows:
                draws.append([idx + 1 for idx, flag in enumerate(row) if flag == 1])
            return [self._validate_draw(row) for row in draws]

        raise ValueError(
            "CSV must contain issue,n1..n20 or 期別,獎號1..20 or 1..80 binary"
        )

    @staticmethod
    def _validate_draw(row: Sequence[int]) -> List[int]:
        values = [int(n) for n in row if 1 <= int(n) <= 80]
        if len(values) != 20 or len(set(values)) != 20:
            raise ValueError("Each draw row must contain 20 unique numbers within 1-80")
        return sorted(values)

    @staticmethod
    def _build_matrix(draw_numbers: Sequence[Sequence[int]]) -> np.ndarray:
        matrix = np.zeros((len(draw_numbers), 80), dtype=np.int8)
        for i, draw in enumerate(draw_numbers):
            for n in draw:
                matrix[i, n - 1] = 1
        return matrix

    @staticmethod
    def _zone_counts(draw: Sequence[int]) -> Dict[str, int]:
        return {
            "A": sum(1 for n in draw if 1 <= n <= 20),
            "B": sum(1 for n in draw if 21 <= n <= 40),
            "C": sum(1 for n in draw if 41 <= n <= 60),
            "D": sum(1 for n in draw if 61 <= n <= 80),
        }

    @staticmethod
    def classify_board(zone_counts: Sequence[int]) -> str:
        max_k = max(zone_counts)
        std_k = float(np.std(zone_counts))
        if max_k >= 7 or std_k > 1.4:
            return "大盤"
        if max_k <= 5 or std_k < 0.9:
            return "小盤"
        return "平均盤"

    @staticmethod
    def _range_counts(draw: Sequence[int]) -> Dict[str, int]:
        return {
            "small": sum(1 for n in draw if 1 <= n <= 26),
            "mid": sum(1 for n in draw if 27 <= n <= 53),
            "large": sum(1 for n in draw if 54 <= n <= 80),
        }

    @staticmethod
    def _consecutive_runs(draw: Sequence[int]) -> List[int]:
        if not draw:
            return []
        draw = sorted(draw)
        runs = []
        run_len = 1
        for i in range(1, len(draw)):
            if draw[i] == draw[i - 1] + 1:
                run_len += 1
            else:
                if run_len >= 2:
                    runs.append(run_len)
                run_len = 1
        if run_len >= 2:
            runs.append(run_len)
        return runs

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        if den == 0:
            return 0.0
        return float(num / den)

    @staticmethod
    def _normalize_weights(
        weights: Dict[str, float], required_keys: Sequence[str]
    ) -> Dict[str, float]:
        merged = {k: float(weights.get(k, 0.0)) for k in required_keys}
        total = float(sum(merged.values()))
        if total <= 0:
            return {k: 1.0 / len(required_keys) for k in required_keys}
        return {k: float(v / total) for k, v in merged.items()}

    @staticmethod
    def _zone_index(number: int) -> int:
        if number <= 20:
            return 0
        if number <= 40:
            return 1
        if number <= 60:
            return 2
        return 3

    @staticmethod
    def _zone_pair_label(zone_indices: Sequence[int]) -> str:
        if len(zone_indices) < 2:
            return "NA"
        return f"{ZONE_LABELS[zone_indices[0]]}-{ZONE_LABELS[zone_indices[1]]}"

    def _draw_structural_features(self, draw: Sequence[int]) -> Dict[str, object]:
        zone_counts_map = self._zone_counts(draw)
        zone_counts = [zone_counts_map[zone] for zone in ZONE_LABELS]
        runs = self._consecutive_runs(draw)
        consecutive_group_count = len(runs)
        max_consecutive_len = max(runs) if runs else 1
        has_consecutive_len_ge4 = int(any(run >= 4 for run in runs))
        is_extreme_dense_consecutive = int(
            consecutive_group_count >= 4 or max_consecutive_len >= 5
        )

        zone_run_presence = [0, 0, 0, 0]
        zone_run_count = [0, 0, 0, 0]
        sorted_draw = sorted(draw)
        run_buffer = [sorted_draw[0]] if sorted_draw else []
        for i in range(1, len(sorted_draw)):
            if sorted_draw[i] == sorted_draw[i - 1] + 1:
                run_buffer.append(sorted_draw[i])
            else:
                if len(run_buffer) >= 2:
                    involved = {self._zone_index(n) for n in run_buffer}
                    for zone_idx in involved:
                        zone_run_presence[zone_idx] = 1
                        zone_run_count[zone_idx] += 1
                run_buffer = [sorted_draw[i]]
        if len(run_buffer) >= 2:
            involved = {self._zone_index(n) for n in run_buffer}
            for zone_idx in involved:
                zone_run_presence[zone_idx] = 1
                zone_run_count[zone_idx] += 1

        strong_zones = [1 if count >= 7 else 0 for count in zone_counts]
        explosive_zones = [1 if count >= 8 else 0 for count in zone_counts]
        zone_std = float(np.std(zone_counts))
        balanced_score = max(0.0, 1.0 - zone_std / 3.0)
        is_balanced = int(
            max(zone_counts) - min(zone_counts) <= 2
            and all(3 <= c <= 7 for c in zone_counts)
        )

        top_two_zones = np.argsort(zone_counts)[::-1][:2].tolist()
        top_two_gap = int(zone_counts[top_two_zones[0]] - zone_counts[top_two_zones[1]])
        is_oscillation = int(zone_counts[top_two_zones[1]] >= 6 and top_two_gap <= 2)
        oscillation_zone_pair = (
            self._zone_pair_label(top_two_zones) if is_oscillation else "NA"
        )

        return {
            "zone_counts": zone_counts,
            "consecutive_group_count": consecutive_group_count,
            "max_consecutive_len": max_consecutive_len,
            "has_consecutive_len_ge4": has_consecutive_len_ge4,
            "is_extreme_dense_consecutive": is_extreme_dense_consecutive,
            "zone_run_presence": zone_run_presence,
            "zone_run_count": zone_run_count,
            "strong_zones": strong_zones,
            "explosive_zones": explosive_zones,
            "strong_zone_count": int(sum(strong_zones)),
            "explosive_zone_count": int(sum(explosive_zones)),
            "is_balanced": is_balanced,
            "balanced_score": round(balanced_score, 6),
            "is_oscillation": is_oscillation,
            "oscillation_zone_pair": oscillation_zone_pair,
            "top_two_zones": top_two_zones,
            "top_two_gap": top_two_gap,
            "dominant_zone": int(np.argmax(zone_counts)),
            # backward compatible aliases
            "run_group_count": consecutive_group_count,
            "max_run": max_consecutive_len,
            "has_ge4": has_consecutive_len_ge4,
            "extreme_dense": is_extreme_dense_consecutive,
            "burst_zones": explosive_zones,
            "balanced": is_balanced,
            "oscillation": is_oscillation,
        }

    def _window_structural_features(
        self, draws: Sequence[Sequence[int]]
    ) -> List[Dict[str, object]]:
        return [self._draw_structural_features(draw) for draw in draws]

    def _window_profile_features(
        self, features: Sequence[Dict[str, object]]
    ) -> Dict[str, object]:
        if not features:
            return {}
        zone_mat = np.array([item["zone_counts"] for item in features], dtype=float)
        strong_ratio = self._safe_div(
            sum(item["strong_zone_count"] for item in features), len(features) * 4
        )
        explosive_ratio = self._safe_div(
            sum(item["explosive_zone_count"] for item in features), len(features) * 4
        )
        balanced_ratio = self._safe_div(
            sum(item["is_balanced"] for item in features), len(features)
        )
        oscillation_ratio = self._safe_div(
            sum(item["is_oscillation"] for item in features), len(features)
        )
        dominant_zone_histogram = [
            sum(1 for item in features if item["dominant_zone"] == i) for i in range(4)
        ]
        pair_counter: Counter = Counter(
            item["oscillation_zone_pair"]
            for item in features
            if item["oscillation_zone_pair"] != "NA"
        )
        return {
            "zone_counts_mean": [float(x) for x in zone_mat.mean(axis=0).tolist()],
            "zone_counts_std": [float(x) for x in zone_mat.std(axis=0).tolist()],
            "strong_ratio": float(strong_ratio),
            "explosive_ratio": float(explosive_ratio),
            "balanced_ratio": float(balanced_ratio),
            "oscillation_ratio": float(oscillation_ratio),
            "max_consecutive_len_mean": float(
                np.mean([item["max_consecutive_len"] for item in features])
            ),
            "dense_consecutive_ratio": float(
                self._safe_div(
                    sum(item["is_extreme_dense_consecutive"] for item in features),
                    len(features),
                )
            ),
            "dominant_zone_histogram": dominant_zone_histogram,
            "dominant_oscillation_pair_histogram": [
                {"pair": str(k), "count": int(v)}
                for k, v in pair_counter.most_common(6)
            ],
        }

    def _trend_profile(
        self, features: Sequence[Dict[str, object]]
    ) -> Dict[str, object]:
        compressed_then_burst = 0
        burst_then_fall = 0
        strong_continuation = 0
        burst_two_streak = 0
        handoff_count = 0
        oscillation_switch = 0
        balanced_then_single_strong = 0
        extreme_consecutive_then_cooldown = 0
        for i in range(1, len(features)):
            prev = features[i - 1]
            cur = features[i]
            if max(prev["zone_counts"]) <= 5 and max(cur["zone_counts"]) >= 8:
                compressed_then_burst += 1
            if max(prev["zone_counts"]) >= 8 and max(cur["zone_counts"]) <= 6:
                burst_then_fall += 1
            if prev["strong_zone_count"] >= 1 and cur["strong_zone_count"] >= 1:
                strong_continuation += 1
            if prev["explosive_zone_count"] >= 1 and cur["explosive_zone_count"] >= 1:
                burst_two_streak += 1
            prev_explosive = {i for i, v in enumerate(prev["explosive_zones"]) if v}
            cur_explosive = {i for i, v in enumerate(cur["explosive_zones"]) if v}
            if (
                prev_explosive
                and cur_explosive
                and prev_explosive.isdisjoint(cur_explosive)
            ):
                handoff_count += 1
            if (
                prev["oscillation_zone_pair"] != "NA"
                and cur["oscillation_zone_pair"] != "NA"
                and prev["oscillation_zone_pair"] != cur["oscillation_zone_pair"]
            ):
                oscillation_switch += 1
            if prev["is_balanced"] and cur["strong_zone_count"] == 1:
                balanced_then_single_strong += 1
            if prev["has_consecutive_len_ge4"] and cur["max_consecutive_len"] <= 2:
                extreme_consecutive_then_cooldown += 1

        profile = self._window_profile_features(features)
        profile.update(
            {
                "compressed_then_burst": compressed_then_burst,
                "burst_then_fall": burst_then_fall,
                "strong_continuation": strong_continuation,
                "burst_two_streak": burst_two_streak,
                "handoff_count": handoff_count,
                "oscillation_switch": oscillation_switch,
                "balanced_then_single_strong": balanced_then_single_strong,
                "extreme_consecutive_then_cooldown": extreme_consecutive_then_cooldown,
                "zone_leader_hist": profile.get(
                    "dominant_zone_histogram", [0, 0, 0, 0]
                ),
                "burst_ratio": profile.get("explosive_ratio", 0.0),
                "strong_ratio": profile.get("strong_ratio", 0.0),
                "balanced_ratio": profile.get("balanced_ratio", 0.0),
                "oscillation_ratio": profile.get("oscillation_ratio", 0.0),
                "oscillation_switches": oscillation_switch,
                "burst_continue": strong_continuation,
                "dominant_oscillation_pairs": profile.get(
                    "dominant_oscillation_pair_histogram", []
                ),
            }
        )
        return profile

    @staticmethod
    def _normalize_vector(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.zeros_like(values, dtype=float)
        max_v = float(values.max())
        min_v = float(values.min())
        if max_v - min_v <= 1e-12:
            return np.zeros_like(values, dtype=float)
        return (values - min_v) / (max_v - min_v)

    def _build_event_effect_table(
        self, historical_features: Sequence[Dict[str, object]]
    ) -> Dict[str, object]:
        tracked_events = [
            "burst_then_fall",
            "handoff_count",
            "balanced_then_single_strong",
            "extreme_consecutive_then_cooldown",
            "oscillation_switch",
        ]
        table = {
            event: {
                "sample_count": 0,
                "zone_delta_mean": [0.0, 0.0, 0.0, 0.0],
                "next_strong_rate": 0.0,
            }
            for event in tracked_events
        }
        accum = {event: [] for event in tracked_events}
        next_strong = {event: 0 for event in tracked_events}

        for i in range(1, len(historical_features) - 1):
            prev = historical_features[i - 1]
            cur = historical_features[i]
            nxt = historical_features[i + 1]
            flags = {
                "burst_then_fall": int(
                    max(prev["zone_counts"]) >= 8 and max(cur["zone_counts"]) <= 6
                ),
                "handoff_count": int(
                    bool(set(idx for idx, v in enumerate(prev["explosive_zones"]) if v))
                    and bool(
                        set(idx for idx, v in enumerate(cur["explosive_zones"]) if v)
                    )
                    and set(
                        idx for idx, v in enumerate(prev["explosive_zones"]) if v
                    ).isdisjoint(
                        set(idx for idx, v in enumerate(cur["explosive_zones"]) if v)
                    )
                ),
                "balanced_then_single_strong": int(
                    cur["is_balanced"] and nxt["strong_zone_count"] == 1
                ),
                "extreme_consecutive_then_cooldown": int(
                    cur["has_consecutive_len_ge4"] and nxt["max_consecutive_len"] <= 2
                ),
                "oscillation_switch": int(
                    prev["oscillation_zone_pair"] != "NA"
                    and cur["oscillation_zone_pair"] != "NA"
                    and prev["oscillation_zone_pair"] != cur["oscillation_zone_pair"]
                ),
            }
            delta = [nxt["zone_counts"][z] - cur["zone_counts"][z] for z in range(4)]
            for event, enabled in flags.items():
                if not enabled:
                    continue
                accum[event].append(delta)
                if nxt["strong_zone_count"] >= 1:
                    next_strong[event] += 1

        for event in tracked_events:
            samples = accum[event]
            if not samples:
                continue
            arr = np.array(samples, dtype=float)
            table[event] = {
                "sample_count": int(len(samples)),
                "zone_delta_mean": [float(v) for v in arr.mean(axis=0).tolist()],
                "next_strong_rate": float(
                    self._safe_div(next_strong[event], len(samples))
                ),
            }
        return table

    def _event_effect_adjustment(
        self,
        current_features: Sequence[Dict[str, object]],
        event_effect_table: Dict[str, object],
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        adjustments = np.zeros(80, dtype=float)
        last = current_features[-1]
        prior = current_features[-2] if len(current_features) >= 2 else last
        active_events = {
            "burst_then_fall": int(
                max(prior["zone_counts"]) >= 8 and max(last["zone_counts"]) <= 6
            ),
            "handoff_count": int(
                bool(set(idx for idx, v in enumerate(prior["explosive_zones"]) if v))
                and bool(set(idx for idx, v in enumerate(last["explosive_zones"]) if v))
                and set(
                    idx for idx, v in enumerate(prior["explosive_zones"]) if v
                ).isdisjoint(
                    set(idx for idx, v in enumerate(last["explosive_zones"]) if v)
                )
            ),
            "balanced_then_single_strong": int(last["is_balanced"]),
            "extreme_consecutive_then_cooldown": int(last["has_consecutive_len_ge4"]),
            "oscillation_switch": int(
                prior["oscillation_zone_pair"] != "NA"
                and last["oscillation_zone_pair"] != "NA"
                and prior["oscillation_zone_pair"] != last["oscillation_zone_pair"]
            ),
        }
        for event, enabled in active_events.items():
            if not enabled:
                continue
            effect = event_effect_table.get(event, {})
            zone_delta = effect.get("zone_delta_mean", [0.0, 0.0, 0.0, 0.0])
            confidence = min(float(effect.get("sample_count", 0)) / 30.0, 1.0)
            for z, delta in enumerate(zone_delta):
                start = z * 20
                end = start + 20
                adjustments[start:end] += delta * 0.06 * confidence
        return adjustments, {
            "active_events": active_events,
            "event_effect_table": event_effect_table,
        }

    def _number_weights_from_group_counts(
        self, counts: Dict[str, int], kind: str = "zone"
    ) -> np.ndarray:
        weights = np.zeros(80, dtype=float)
        for n in range(1, 81):
            if kind == "zone":
                key = "A" if n <= 20 else "B" if n <= 40 else "C" if n <= 60 else "D"
            else:
                key = "small" if n <= 26 else "mid" if n <= 53 else "large"
            weights[n - 1] = counts[key] / 20
        return self._normalize_vector(weights)

    def _tail_digit_stats(self, draws: Sequence[Sequence[int]]) -> Dict[int, int]:
        counter: Counter = Counter()
        for draw in draws:
            for n in draw:
                counter[n % 10] += 1
        return {d: int(counter[d]) for d in range(10)}

    def _fixed_gap_stats(
        self, draws: Sequence[Sequence[int]], gaps: Sequence[int] = (5, 10, 20, 30)
    ) -> Dict[int, int]:
        out = {int(g): 0 for g in gaps}
        for draw in draws:
            s = sorted(draw)
            for i, j in combinations(s, 2):
                diff = j - i
                if diff in out:
                    out[diff] += 1
        return out

    def _skip_pattern_stats(self, draws: Sequence[Sequence[int]]) -> Dict[int, int]:
        counter = Counter()
        for draw in draws:
            s = sorted(set(draw))
            for a, b, c in combinations(s, 3):
                if b - a == c - b and b - a >= 2:
                    counter[b - a] += 1
        return {int(k): int(v) for k, v in counter.items()}

    def _consecutive_pattern_tables(
        self, draws: Sequence[Sequence[int]]
    ) -> Dict[str, List[Dict[str, object]]]:
        pair_counter: Counter = Counter()
        triple_counter: Counter = Counter()
        for draw in draws:
            s = sorted(draw)
            for i in range(len(s) - 1):
                if s[i + 1] == s[i] + 1:
                    pair_counter[(s[i], s[i + 1])] += 1
            for i in range(len(s) - 2):
                if s[i + 1] == s[i] + 1 and s[i + 2] == s[i + 1] + 1:
                    triple_counter[(s[i], s[i + 1], s[i + 2])] += 1
        return {
            "pairs": [
                {"numbers": list(k), "count": int(v)}
                for k, v in pair_counter.most_common(20)
            ],
            "triples": [
                {"numbers": list(k), "count": int(v)}
                for k, v in triple_counter.most_common(20)
            ],
        }

    def _build_feature_snapshot(
        self, draws: Sequence[Sequence[int]]
    ) -> Dict[str, object]:
        zone_df = pd.DataFrame([self._zone_counts(d) for d in draws])
        range_df = pd.DataFrame([self._range_counts(d) for d in draws])
        tail = self._tail_digit_stats(draws)
        gaps = self._fixed_gap_stats(draws)
        skip = self._skip_pattern_stats(draws)
        consecutive = self._consecutive_pattern_tables(draws)
        return {
            "zone_mean": {k: float(v) for k, v in zone_df.mean().to_dict().items()},
            "range_mean": {k: float(v) for k, v in range_df.mean().to_dict().items()},
            "tail": tail,
            "gaps": gaps,
            "skip": skip,
            "consecutive": consecutive,
        }

    def _detect_pattern_spikes(
        self,
        recent_draws: Sequence[Sequence[int]],
        baseline_draws: Sequence[Sequence[int]],
    ) -> Dict[str, bool]:
        recent_zone = pd.DataFrame([self._zone_counts(d) for d in recent_draws]).mean()
        base_zone = pd.DataFrame([self._zone_counts(d) for d in baseline_draws]).mean()
        zone_burst = bool(((recent_zone - base_zone).abs() >= 1.0).any())

        recent_tail = np.array(
            list(self._tail_digit_stats(recent_draws).values()), dtype=float
        )
        base_tail = np.array(
            list(self._tail_digit_stats(baseline_draws).values()), dtype=float
        )
        tail_cluster = bool(
            recent_tail.max() > (base_tail.max() * 1.2 if base_tail.max() > 0 else 0)
        )

        recent_cons = len(self._consecutive_pattern_tables(recent_draws)["pairs"])
        base_cons = max(
            1, len(self._consecutive_pattern_tables(baseline_draws)["pairs"])
        )
        consecutive_spike = bool(recent_cons > base_cons * 1.2)
        return {
            "zone_burst": zone_burst,
            "tail_cluster": tail_cluster,
            "consecutive_spike": consecutive_spike,
            "cluster_burst": bool(zone_burst or tail_cluster or consecutive_spike),
        }

    def _adaptive_weights(
        self, spikes: Dict[str, bool], cluster_score: float = 0.0
    ) -> Dict[str, float]:
        weights = ScoreWeights().as_dict()
        if spikes.get("zone_burst"):
            weights["zone_distribution"] += 0.05
        if spikes.get("tail_cluster"):
            weights["tail_concentration"] += 0.03
        if spikes.get("consecutive_spike"):
            weights["consecutive_pattern"] += 0.03
        if spikes.get("cluster_burst"):
            weights["cluster_pattern"] = 0.07 + (0.02 * min(float(cluster_score), 5.0))

        total = float(sum(weights.values()))
        return {k: float(v / total) for k, v in weights.items()}

    def _cluster_burst_analysis(
        self,
        draws: Sequence[Sequence[int]],
        window: int = 10,
    ) -> Tuple[np.ndarray, List[List[int]], Dict[str, object]]:
        recent_draws = list(draws)[-min(window, len(draws)) :]
        if not recent_draws:
            return (
                np.zeros(80, dtype=float),
                [],
                {
                    "window": 0,
                    "interval_cluster": 0,
                    "tail_cluster": 0,
                    "consecutive_cluster": 0,
                },
            )

        interval_hits = [0, 0, 0, 0]
        tail_hits: Counter = Counter()
        consecutive_number_hits: Counter = Counter()
        consecutive_group_pool: List[List[int]] = []

        for draw in recent_draws:
            zone_counts = self._zone_counts(draw)
            for idx, zone in enumerate(ZONE_LABELS):
                if zone_counts[zone] >= 7:
                    interval_hits[idx] += 1

            tail_counter = Counter(n % 10 for n in draw)
            for tail, count in tail_counter.items():
                if count >= 3:
                    tail_hits[tail] += 1

            sorted_draw = sorted(draw)
            run_groups: List[List[int]] = []
            current_run = [sorted_draw[0]]
            for i in range(1, len(sorted_draw)):
                if sorted_draw[i] == sorted_draw[i - 1] + 1:
                    current_run.append(sorted_draw[i])
                else:
                    if len(current_run) >= 2:
                        run_groups.append(current_run[:])
                    current_run = [sorted_draw[i]]
            if len(current_run) >= 2:
                run_groups.append(current_run)

            if len(run_groups) >= 2:
                for group in run_groups:
                    consecutive_group_pool.append(group)
                    for number in group:
                        consecutive_number_hits[number] += 1

        interval_component = np.zeros(80, dtype=float)
        for zone_idx, hit in enumerate(interval_hits):
            if hit <= 0:
                continue
            start = zone_idx * 20
            interval_component[start : start + 20] = float(hit)

        tail_component = np.array(
            [float(tail_hits[(i + 1) % 10]) for i in range(80)],
            dtype=float,
        )
        consecutive_component = np.array(
            [float(consecutive_number_hits.get(i + 1, 0)) for i in range(80)],
            dtype=float,
        )

        cluster_raw = interval_component + tail_component + consecutive_component
        cluster_component = self._normalize_vector(cluster_raw)

        cluster_groups: List[List[int]] = []
        for group in sorted(consecutive_group_pool, key=lambda x: (-len(x), x[0])):
            if len(group) < 3:
                continue
            if group not in cluster_groups:
                cluster_groups.append(group)
            if len(cluster_groups) >= 5:
                break

        if not cluster_groups:
            for tail, hit in tail_hits.most_common(2):
                if hit <= 0:
                    continue
                group = [n for n in range(tail if tail > 0 else 10, 81, 10)]
                if len(group) >= 3:
                    cluster_groups.append(group)

        if not cluster_groups:
            for zone_idx, hit in sorted(
                enumerate(interval_hits),
                key=lambda x: x[1],
                reverse=True,
            ):
                if hit <= 0:
                    continue
                start = zone_idx * 20 + 1
                cluster_groups.append(list(range(start, min(start + 5, start + 20))))
                if len(cluster_groups) >= 2:
                    break

        metadata = {
            "window": len(recent_draws),
            "interval_cluster": int(sum(interval_hits)),
            "tail_cluster": int(sum(tail_hits.values())),
            "consecutive_cluster": int(sum(consecutive_number_hits.values())),
            "interval_zones": {
                zone: int(interval_hits[idx]) for idx, zone in enumerate(ZONE_LABELS)
            },
            "tail_digits": {str(t): int(c) for t, c in sorted(tail_hits.items())},
        }
        return cluster_component, cluster_groups, metadata

    def _history_pattern_similarity_component(
        self,
        recent_draws: Sequence[Sequence[int]],
        latest_issue: int,
        sequence_window_size: int = 10,
        top_n: int = 500,
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        if not recent_draws:
            return np.zeros(80, dtype=float), []

        window_size = min(max(sequence_window_size, 1), len(recent_draws))
        query_sequence = [sorted(draw) for draw in list(recent_draws)[-window_size:]]
        query_zone_seq = [self._zone_counts(draw) for draw in query_sequence]
        query_range_seq = [self._range_counts(draw) for draw in query_sequence]
        step_weights = np.arange(1, window_size + 1, dtype=float)
        step_weights = step_weights / step_weights.sum()

        coarse_candidates: List[Tuple[float, int, int]] = []
        detailed_candidates: List[Tuple[float, float, int, int, int]] = []
        latest_index = self.issue_to_index.get(latest_issue)
        search_end_index = (
            latest_index if latest_index is not None else len(self.draw_numbers)
        )
        max_start_exclusive = max(search_end_index - window_size, 0)
        search_start_index = max(0, max_start_exclusive - 1200)
        for start_idx in range(search_start_index, max_start_exclusive):
            end_idx = start_idx + window_size - 1
            next_idx = end_idx + 1
            if next_idx >= search_end_index:
                continue

            hist_sequence = self.draw_numbers[start_idx : end_idx + 1]
            coarse_sim = 0.0
            overlap_sim = 0.0
            consecutive_sim = 0.0
            for pos in range(window_size):
                hist_draw = hist_sequence[pos]
                query_draw = query_sequence[pos]
                hist_zone = self._zone_counts(hist_draw)
                hist_range = self._range_counts(hist_draw)

                zone_sim = 1 - (
                    sum(abs(hist_zone[k] - query_zone_seq[pos][k]) for k in hist_zone)
                    / 40
                )
                range_sim = 1 - (
                    sum(
                        abs(hist_range[k] - query_range_seq[pos][k]) for k in hist_range
                    )
                    / 40
                )
                set_sim = len(set(hist_draw) & set(query_draw)) / len(
                    set(hist_draw) | set(query_draw)
                )
                hist_runs = set(self._consecutive_runs(hist_draw))
                query_runs = set(self._consecutive_runs(query_draw))
                run_union = len(hist_runs | query_runs)
                run_sim = (
                    1.0 if run_union == 0 else len(hist_runs & query_runs) / run_union
                )

                coarse_sim += step_weights[pos] * (0.60 * zone_sim + 0.40 * range_sim)
                overlap_sim += step_weights[pos] * set_sim
                consecutive_sim += step_weights[pos] * run_sim

            start_issue = int(self.df.iloc[start_idx]["issue"])
            end_issue = int(self.df.iloc[end_idx]["issue"])
            next_issue = int(self.df.iloc[next_idx]["issue"])
            coarse_candidates.append((float(coarse_sim), start_idx, next_issue))
            detailed_candidates.append(
                (
                    float(0.70 * overlap_sim + 0.30 * consecutive_sim),
                    float(coarse_sim),
                    start_issue,
                    end_issue,
                    next_issue,
                )
            )

        coarse_candidates.sort(key=lambda x: x[0], reverse=True)
        coarse_allowed_next_issue = {
            next_issue for _, _, next_issue in coarse_candidates[:1000]
        }
        filtered_candidates = [
            item for item in detailed_candidates if item[4] in coarse_allowed_next_issue
        ]
        filtered_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        selected = [item for item in filtered_candidates if item[0] > 0][:top_n]
        history_scores = np.zeros(80, dtype=float)
        details: List[Dict[str, object]] = []
        total_weight = sum(x[0] for x in selected)
        if total_weight <= 0:
            return history_scores, details
        for sim, _, start_issue, end_issue, next_issue in selected:
            next_draw = self.draw_numbers[self.issue_to_index[next_issue]]
            details.append(
                {
                    "sequence_start_issue": int(start_issue),
                    "sequence_end_issue": int(end_issue),
                    "next_issue": int(next_issue),
                    "sequence_window_size": int(window_size),
                    "similarity": round(float(sim), 6),
                }
            )
            for n in next_draw:
                history_scores[n - 1] += sim
        return history_scores / total_weight, details

    def _blended_tail_component(
        self,
        recent_draws: Sequence[Sequence[int]],
        baseline_window: int = 30,
    ) -> np.ndarray:
        recent_tail = self._tail_digit_stats(recent_draws)
        baseline_tail = self._tail_digit_stats(self.draw_numbers[-baseline_window:])
        tail_scores = np.array(
            [
                0.5 * recent_tail[n % 10] + 0.5 * baseline_tail[n % 10]
                for n in range(1, 81)
            ],
            dtype=float,
        )
        return self._normalize_vector(tail_scores)

    def _gap_skip_hotspot_component(
        self, recent_draws: Sequence[Sequence[int]]
    ) -> np.ndarray:
        gap_stats = self._fixed_gap_stats(recent_draws)
        skip_stats = self._skip_pattern_stats(recent_draws)
        hotspot_numbers = {
            number
            for number in range(1, 81)
            if gap_stats.get(number, 0) > 0 or skip_stats.get(number, 0) > 0
        }
        hotspot_component = np.zeros(80, dtype=float)
        for idx, number in enumerate(range(1, 81)):
            boost = 0.0
            for shift in (10, 20):
                if (
                    number - shift in hotspot_numbers
                    or number + shift in hotspot_numbers
                ):
                    boost = max(boost, 0.06 if shift == 10 else 0.04)
            hotspot_component[idx] = boost
        return self._normalize_vector(hotspot_component)

    def _sum_component(self, draws: Sequence[Sequence[int]]) -> np.ndarray:
        history_draws = list(self.draw_numbers[-800:])
        if not history_draws:
            return np.zeros(80, dtype=float)
        sum_values = np.array([sum(draw) for draw in history_draws], dtype=float)
        hist, bin_edges = np.histogram(sum_values, bins=20)
        if not hist.size:
            return np.zeros(80, dtype=float)
        mode_idx = int(np.argmax(hist))
        lower_idx = max(0, mode_idx - 1)
        upper_idx = min(len(bin_edges) - 1, mode_idx + 2)
        sum_low = float(bin_edges[lower_idx])
        sum_high = float(bin_edges[upper_idx])

        recent_ref = list(draws)[-30:] if draws else self.recent_30_draws
        if not recent_ref:
            return np.zeros(80, dtype=float)
        recent_mean_sum = float(np.mean([sum(draw) for draw in recent_ref]))
        recent_mean_number = float(np.mean([n for draw in recent_ref for n in draw]))

        component = np.zeros(80, dtype=float)
        for idx, number in enumerate(range(1, 81)):
            projected_sum = recent_mean_sum + (number - recent_mean_number)
            if sum_low <= projected_sum <= sum_high:
                component[idx] = 1.0
        return self._normalize_vector(component)

    def _balance_component(self, target: Optional[Dict[str, int]] = None) -> np.ndarray:
        balance_target = target or {"odd": 10, "even": 10, "high": 10, "low": 10}
        reference_draw = self.recent_30_draws[-1] if self.recent_30_draws else []
        odd_count = sum(1 for n in reference_draw if n % 2 == 1)
        even_count = len(reference_draw) - odd_count
        high_count = sum(1 for n in reference_draw if n > 40)
        low_count = len(reference_draw) - high_count
        deficits = {
            "odd": balance_target.get("odd", 10) - odd_count,
            "even": balance_target.get("even", 10) - even_count,
            "high": balance_target.get("high", 10) - high_count,
            "low": balance_target.get("low", 10) - low_count,
        }

        component = np.zeros(80, dtype=float)
        for idx, number in enumerate(range(1, 81)):
            score = 0.01
            score += (
                max(deficits["odd"], 0) if number % 2 == 1 else max(deficits["even"], 0)
            )
            score += (
                max(deficits["high"], 0) if number > 40 else max(deficits["low"], 0)
            )
            component[idx] = float(score)
        return self._normalize_vector(component)

    def _delta_component(self, draws: Sequence[Sequence[int]]) -> np.ndarray:
        if not draws:
            return np.zeros(80, dtype=float)
        recent_matrix = self._build_matrix(draws)
        hot_indices = np.argsort(recent_matrix.sum(axis=0))[::-1][:10]
        hot_numbers = sorted(int(idx + 1) for idx in hot_indices)
        if len(hot_numbers) < 2:
            return np.zeros(80, dtype=float)

        delta_counter: Counter = Counter()
        for a, b in combinations(hot_numbers, 2):
            delta_counter[b - a] += 1
        top_deltas = [delta for delta, _ in delta_counter.most_common(10)]
        hot_set = set(hot_numbers)
        component = np.zeros(80, dtype=float)
        for idx, number in enumerate(range(1, 81)):
            delta_score = 0.0
            for delta in top_deltas:
                if (number - delta) in hot_set or (number + delta) in hot_set:
                    delta_score += float(delta_counter[delta])
            component[idx] = delta_score
        return self._normalize_vector(component)

    def _skip_component(self) -> np.ndarray:
        component = np.zeros(80, dtype=float)
        for number in range(1, 81):
            skip = 9999
            for offset, draw in enumerate(reversed(self.draw_numbers)):
                if number in draw:
                    skip = offset
                    break
            if skip == 0:
                component[number - 1] = 0.01
            elif 1 <= skip <= 5:
                component[number - 1] = 0.03
        return self._normalize_vector(component)

    def _prime_component(self) -> np.ndarray:
        component = np.array(
            [1.0 if number in PRIME_SET else 0.0 for number in range(1, 81)],
            dtype=float,
        )
        return self._normalize_vector(component)

    def _compression_component(self, draws: Sequence[Sequence[int]]) -> np.ndarray:
        recent_draws = list(draws)[-3:]
        if len(recent_draws) < 3:
            return np.zeros(80, dtype=float)
        zone_counters = [self._zone_counts(draw) for draw in recent_draws]
        compressed_zones = [
            zone
            for zone in ZONE_LABELS
            if all(zone_count[zone] <= 4 for zone_count in zone_counters)
        ]
        component = np.zeros(80, dtype=float)
        for zone in compressed_zones:
            start = ZONE_LABELS.index(zone) * 20
            component[start : start + 20] = 1.0
        return self._normalize_vector(component)

    def _resolve_last_draw_penalty(
        self, recent_draws: Sequence[Sequence[int]]
    ) -> float:
        recent_three = list(recent_draws)[-3:]
        if len(recent_three) < 3:
            return DEFAULT_LAST_DRAW_PENALTY
        for zone in ZONE_LABELS:
            if all(self._zone_counts(draw)[zone] >= 7 for draw in recent_three):
                return RELAXED_LAST_DRAW_PENALTY
        return DEFAULT_LAST_DRAW_PENALTY

    def _combo_resonance_scores(
        self, recent_draws: Sequence[Sequence[int]]
    ) -> Tuple[np.ndarray, Counter]:
        pair_counter = Counter()
        triplet_counter = Counter()
        for draw in recent_draws:
            for pair in combinations(draw, 2):
                pair_counter[tuple(sorted(pair))] += 1
            for triplet in combinations(draw, 3):
                triplet_counter[tuple(sorted(triplet))] += 1

        scores = np.zeros(80, dtype=float)
        norm = max(pair_counter.values(), default=1)
        for (a, b), cnt in pair_counter.items():
            val = cnt / norm
            scores[a - 1] += val
            scores[b - 1] += val

        if scores.max() > 0:
            scores = scores / scores.max()
        return scores, triplet_counter

    def _top_same_draw_combinations(
        self,
        candidate_numbers: Sequence[int],
        score: np.ndarray,
        triplet_counter: Counter,
        top_n: int = 3,
        pool_size: int = 12,
    ) -> List[List[int]]:
        pool = list(candidate_numbers[: min(pool_size, len(candidate_numbers))])
        max_count = max(triplet_counter.values(), default=1)
        ranked: List[Tuple[float, Tuple[int, int, int]]] = []
        for combo in combinations(pool, 3):
            combo = tuple(sorted(combo))
            support = triplet_counter.get(combo, 0) / max_count
            avg_score = float(np.mean([score[n - 1] for n in combo]))
            ranked.append((0.6 * avg_score + 0.4 * support, combo))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [list(item[1]) for item in ranked[:top_n]]

    @staticmethod
    def _validate_recent_draws(recent_draws: Sequence[Sequence[int]]) -> None:
        if not 10 <= len(recent_draws) <= 50:
            raise ValueError(PREDICT_REQUIRED_MESSAGE)
        for draw in recent_draws:
            if len(draw) != 20:
                raise ValueError("recent 每期 numbers 必須恰好 20 顆。")
            if len(set(draw)) != 20:
                raise ValueError("recent 每期 numbers 不可重複。")
            if any((n < 1 or n > 80) for n in draw):
                raise ValueError("recent 每期 numbers 必須介於 1 到 80。")

    @staticmethod
    def _normalize_target_counts(
        target: Dict[str, int], total: int = 20
    ) -> Dict[str, int]:
        keys = list(target.keys())
        current = sum(target.values())
        while current > total:
            k = max(keys, key=lambda x: target[x])
            target[k] -= 1
            current -= 1
        while current < total:
            k = min(keys, key=lambda x: target[x])
            target[k] += 1
            current += 1
        return target

    def _predict_distribution_target(
        self,
        recent_draws: Sequence[Sequence[int]],
        mode: str,
    ) -> Dict[str, int]:
        if mode == "zone":
            latest = self._zone_counts(recent_draws[-1])
            short_means = pd.DataFrame(
                [self._zone_counts(d) for d in recent_draws[-10:]]
            ).mean()
            long_means = pd.DataFrame(
                [self._zone_counts(d) for d in self.draw_numbers[-200:]]
            ).mean()
            target = {
                k: int(
                    round(
                        (0.15 * latest[k])
                        + (0.35 * short_means[k])
                        + (0.50 * long_means[k])
                    )
                )
                for k in latest
            }
        else:
            latest = self._range_counts(recent_draws[-1])
            short_means = pd.DataFrame(
                [self._range_counts(d) for d in recent_draws[-10:]]
            ).mean()
            long_means = pd.DataFrame(
                [self._range_counts(d) for d in self.draw_numbers[-200:]]
            ).mean()
            target = {
                k: int(
                    round(
                        (0.15 * latest[k])
                        + (0.35 * short_means[k])
                        + (0.50 * long_means[k])
                    )
                )
                for k in latest
            }
        return self._normalize_target_counts(target)

    @staticmethod
    def _apply_last_draw_penalty(
        score: np.ndarray, last_draw: Sequence[int], penalty_factor: float
    ) -> np.ndarray:
        penalized = score.copy()
        for number in last_draw:
            penalized[number - 1] *= penalty_factor
        return penalized

    @staticmethod
    def _select_with_last_draw_cap(
        ranking: Sequence[int],
        last_draw: Sequence[int],
        top_k: int,
        max_last_draw: int = DEFAULT_LAST_DRAW_MAX_IN_TOPK,
    ) -> List[int]:
        selected: List[int] = []
        last_draw_set = set(last_draw)
        last_draw_count = 0
        deferred: List[int] = []

        for number in ranking:
            if len(selected) >= top_k:
                break
            if number in last_draw_set and last_draw_count >= max_last_draw:
                deferred.append(number)
                continue
            selected.append(int(number))
            if number in last_draw_set:
                last_draw_count += 1

        if len(selected) < top_k:
            for number in deferred:
                if len(selected) >= top_k:
                    break
                selected.append(int(number))

        return selected

    def _momentum_scores(
        self, draws: Sequence[Sequence[int]], window: int
    ) -> np.ndarray:
        sub_draws = list(draws)[-window:]
        sub_matrix = self._build_matrix(sub_draws)
        return sub_matrix.sum(axis=0) / max(len(sub_draws), 1)

    def _hot_frequency_scores(self) -> np.ndarray:
        return self.matrix.mean(axis=0)

    def basic_statistics(self, top_n_triplets: int = 10) -> Dict[str, object]:
        total_draws = len(self.draw_numbers)
        counts = self.matrix.sum(axis=0)
        probs = counts / max(total_draws, 1)

        zone_per_draw = pd.DataFrame([self._zone_counts(d) for d in self.draw_numbers])
        zone_avg = zone_per_draw.mean().to_dict()
        zone_burst_ge7 = (zone_per_draw >= 7).sum().to_dict()
        zone_burst_ge8 = (zone_per_draw >= 8).sum().to_dict()

        range_per_draw = pd.DataFrame(
            [self._range_counts(d) for d in self.draw_numbers]
        )
        range_avg = range_per_draw.mean().to_dict()
        triplets = Counter()
        triplet_draws = self.draw_numbers[-min(total_draws, 1200) :]
        for draw in triplet_draws:
            for combo in combinations(draw, 3):
                triplets[combo] += 1

        return {
            "total_draws": total_draws,
            "number_total_counts": {str(i + 1): int(c) for i, c in enumerate(counts)},
            "number_probabilities": {str(i + 1): float(p) for i, p in enumerate(probs)},
            "zone_stats": {
                "average_per_draw": {k: float(v) for k, v in zone_avg.items()},
                "burst_ge_7": {k: int(v) for k, v in zone_burst_ge7.items()},
                "burst_ge_8": {k: int(v) for k, v in zone_burst_ge8.items()},
            },
            "big_mid_small_stats": {
                "average_per_draw": {k: float(v) for k, v in range_avg.items()}
            },
            "top_triplets": [
                {"numbers": list(nums), "count": c}
                for nums, c in triplets.most_common(top_n_triplets)
            ],
            "history_verification": self.history_verification,
        }

    def dynamic_analysis(
        self,
        recent_draws: Optional[Sequence[Sequence[int]]] = None,
        latest_issue: Optional[int] = None,
    ) -> Dict[str, object]:
        draws = list(recent_draws) if recent_draws is not None else self.draw_numbers
        latest_draw = draws[-1]
        resolved_latest_issue = (
            latest_issue if latest_issue is not None else int(self.df.iloc[-1]["issue"])
        )
        recent_window = draws[-50:]
        baseline = self.draw_numbers[
            : max(1, len(self.draw_numbers) - len(recent_window))
        ]
        return {
            "latest_issue": resolved_latest_issue,
            "latest_draw": list(latest_draw),
            "recent_features": self._build_feature_snapshot(recent_window),
            "history_features": self._build_feature_snapshot(baseline),
            "history_verification": self._verify_history_usage(len(recent_window)),
        }

    def predict_next(
        self,
        recent_draws: Sequence[Sequence[int]],
        latest_issue: int,
        top_k: int = 20,
        alpha: float = 1.0,
        lambda_value: float = 1.0,
        candidate_pool_size: int = 12,
    ) -> Dict[str, object]:
        self._validate_recent_draws(recent_draws)
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        if lambda_value < 0:
            raise ValueError("lambda must be >= 0.")
        short_window = len(recent_draws)
        target_issue = latest_issue + 1
        recent_matrix = self._build_matrix(recent_draws)
        recency_weights = np.array(
            [alpha ** (short_window - 1 - idx) for idx in range(short_window)],
            dtype=float,
        )
        recency_weights = recency_weights / recency_weights.sum()
        recent_freq = (recent_matrix * recency_weights[:, None]).sum(axis=0)

        baseline_draws = self.draw_numbers[
            : max(1, len(self.draw_numbers) - short_window)
        ]
        spikes = self._detect_pattern_spikes(recent_draws, baseline_draws)

        zone_target = self._predict_distribution_target(recent_draws, mode="zone")
        range_target = self._predict_distribution_target(recent_draws, mode="range")

        zone_component = self._number_weights_from_group_counts(
            zone_target, kind="zone"
        )
        range_component = self._number_weights_from_group_counts(
            range_target, kind="range"
        )
        recent_component = self._normalize_vector(recent_freq)
        hot_component = self._normalize_vector(self._hot_frequency_scores())

        sequence_window_size = min(10, max(5, len(recent_draws)))
        history_component, similar_cases = self._history_pattern_similarity_component(
            recent_draws=recent_draws,
            latest_issue=latest_issue,
            sequence_window_size=sequence_window_size,
        )
        history_component = self._normalize_vector(history_component)

        tail_component = self._blended_tail_component(recent_draws)

        cons = self._consecutive_pattern_tables(recent_draws)
        consecutive_numbers = {
            n for item in cons["pairs"][:30] for n in item["numbers"]
        }
        cons_component = np.array(
            [1.0 if (i + 1) in consecutive_numbers else 0.0 for i in range(80)]
        )
        cons_component = self._normalize_vector(cons_component)

        cluster_component, cluster_groups, cluster_metadata = (
            self._cluster_burst_analysis(
                recent_draws,
                window=min(10, short_window),
            )
        )
        cluster_score = (
            cluster_metadata["interval_cluster"]
            + cluster_metadata["tail_cluster"]
            + cluster_metadata["consecutive_cluster"]
        ) / max(cluster_metadata["window"], 1)
        adaptive_weights = self._adaptive_weights(spikes, cluster_score=cluster_score)

        gap_component = self._gap_skip_hotspot_component(recent_draws)
        sum_component = self._sum_component(recent_draws)
        balance_component = self._balance_component()
        delta_component = self._delta_component(recent_draws)
        skip_component = self._skip_component()
        prime_component = self._prime_component()
        compression_component = self._compression_component(recent_draws)

        pattern_component = self._normalize_vector(
            (history_component + (lambda_value * hot_component)) / (1.0 + lambda_value)
        )

        latest_zone_counts_map = self._zone_counts(recent_draws[-1])
        latest_zone_counts = [latest_zone_counts_map[zone] for zone in ZONE_LABELS]
        board_type = self.classify_board(latest_zone_counts)
        recent_w, hist_w, other_w = WEIGHT_TABLE[board_type]

        other_weights = {
            key: value
            for key, value in adaptive_weights.items()
            if key not in {"recent_momentum", "pattern_similarity"}
        }
        other_total = float(sum(other_weights.values()))
        if other_total <= 0:
            other_total = 1.0
        other_component = self._normalize_vector(
            (
                other_weights["zone_distribution"] * zone_component
                + other_weights["hot_frequency"] * hot_component
                + other_weights["big_mid_small"] * range_component
                + other_weights["consecutive_pattern"] * cons_component
                + other_weights["cluster_pattern"] * cluster_component
                + other_weights["tail_concentration"] * tail_component
                + other_weights["gap_skip_pattern"] * gap_component
                + other_weights["sum_range"] * sum_component
                + other_weights["odd_even_balance"] * balance_component
                + other_weights["delta_pattern"] * delta_component
                + other_weights["skip_heat"] * skip_component
                + other_weights["prime_boost"] * prime_component
                + other_weights["compression_boost"] * compression_component
            )
            / other_total
        )

        score = (
            recent_w * recent_component
            + hist_w * pattern_component
            + other_w * other_component
        )

        penalty_factor = self._resolve_last_draw_penalty(recent_draws)
        penalized_score = self._apply_last_draw_penalty(
            score, recent_draws[-1], penalty_factor
        )
        ranking = (np.argsort(penalized_score)[::-1] + 1).tolist()
        selected = self._select_with_last_draw_cap(
            ranking,
            recent_draws[-1],
            top_k=top_k,
        )
        top10 = self._select_with_last_draw_cap(
            ranking,
            recent_draws[-1],
            top_k=10,
        )
        _, triplet_counter = self._combo_resonance_scores(recent_draws)
        top3_combos = self._top_same_draw_combinations(
            selected,
            penalized_score,
            triplet_counter,
            top_n=3,
            pool_size=candidate_pool_size,
        )

        explanation = {
            "zone_burst": "detected" if spikes["zone_burst"] else "not_detected",
            "tail_cluster": "detected" if spikes["tail_cluster"] else "not_detected",
            "consecutive_trend": (
                "detected" if spikes["consecutive_spike"] else "not_detected"
            ),
            "cluster_burst": (
                "detected" if spikes["cluster_burst"] else "not_detected"
            ),
            "cluster_score": {
                "interval_cluster": cluster_metadata["interval_cluster"],
                "tail_cluster": cluster_metadata["tail_cluster"],
                "consecutive_cluster": cluster_metadata["consecutive_cluster"],
            },
            "cluster_analysis": cluster_metadata,
            "weights": adaptive_weights,
            "rule_based_board_type": board_type,
            "rule_based_zone_counts": {
                zone: latest_zone_counts_map[zone] for zone in ZONE_LABELS
            },
            "rule_based_score_weights": {
                "recent": recent_w,
                "history": hist_w,
                "other": other_w,
            },
            "similar_cases_used": len(similar_cases),
            "sequence_similarity_window_size": sequence_window_size,
            "last_draw_penalty": penalty_factor,
            "last_draw_max_in_topk": min(DEFAULT_LAST_DRAW_MAX_IN_TOPK, top_k),
            "last_draw_overlap_in_prediction": len(
                set(selected) & set(recent_draws[-1])
            ),
        }

        return {
            "latest_issue": latest_issue,
            "target_issue": target_issue,
            "short_window": short_window,
            "predicted_zone_distribution": zone_target,
            "predicted_zone_counts": zone_target,
            "predicted_big_mid_small_distribution": range_target,
            "predicted_big_mid_small": range_target,
            "top_10_candidate_numbers": top10,
            "top10_numbers": top10,
            "predicted_numbers_top20": selected,
            "cluster_groups": cluster_groups,
            "top_3_same_draw_combinations": top3_combos,
            "top3_combinations": top3_combos,
            "top3_triplet": {
                "numbers": top3_combos[0],
                "score": float(
                    np.mean([penalized_score[n - 1] for n in top3_combos[0]])
                ),
            },
            "explanation_of_influential_patterns": explanation,
            "explanation": explanation,
            "weights": {
                "base": ScoreWeights().as_dict(),
                "adaptive": adaptive_weights,
            },
            "history_verification": self._verify_history_usage(short_window),
        }

    def _sequence_similarity_components(
        self,
        current_features: Sequence[Dict[str, object]],
        historical_features: Sequence[Dict[str, object]],
        current_draws: Sequence[Sequence[int]],
        historical_draws: Sequence[Sequence[int]],
    ) -> Dict[str, float]:
        overlap_scores = []
        zone_scores = []
        consecutive_scores = []
        for current_draw, hist_draw, current_f, hist_f in zip(
            current_draws,
            historical_draws,
            current_features,
            historical_features,
        ):
            overlap_scores.append(len(set(current_draw) & set(hist_draw)) / 20.0)
            zone_gap = sum(
                abs(a - b)
                for a, b in zip(current_f["zone_counts"], hist_f["zone_counts"])
            )
            zone_scores.append(1 - zone_gap / 40.0)

            run_gap = min(
                abs(
                    current_f["consecutive_group_count"]
                    - hist_f["consecutive_group_count"]
                )
                / 10.0,
                1.0,
            )
            max_run_gap = min(
                abs(current_f["max_consecutive_len"] - hist_f["max_consecutive_len"])
                / 10.0,
                1.0,
            )
            ge4_gap = abs(
                current_f["has_consecutive_len_ge4"] - hist_f["has_consecutive_len_ge4"]
            )
            dense_gap = abs(
                current_f["is_extreme_dense_consecutive"]
                - hist_f["is_extreme_dense_consecutive"]
            )
            zone_run_gap = (
                sum(
                    abs(a - b)
                    for a, b in zip(
                        current_f["zone_run_presence"], hist_f["zone_run_presence"]
                    )
                )
                / 4.0
            )
            consecutive_scores.append(
                1
                - min(
                    1.0,
                    0.25 * run_gap
                    + 0.30 * max_run_gap
                    + 0.20 * ge4_gap
                    + 0.15 * dense_gap
                    + 0.10 * zone_run_gap,
                )
            )

        current_profile = self._trend_profile(current_features)
        historical_profile = self._trend_profile(historical_features)

        trend_key_weights = {
            "compressed_then_burst": 0.20,
            "burst_then_fall": 0.20,
            "strong_continuation": 0.15,
            "burst_two_streak": 0.15,
            "handoff_count": 0.15,
            "oscillation_switch": 0.15,
        }
        trend_score = 0.0
        for key, weight in trend_key_weights.items():
            base = max(len(current_features) - 1, 1)
            diff_ratio = min(
                abs(current_profile[key] - historical_profile[key]) / base, 1.0
            )
            trend_score += weight * (1 - diff_ratio)

        board_key_weights = {
            "explosive_ratio": 0.25,
            "strong_ratio": 0.20,
            "balanced_ratio": 0.25,
            "oscillation_ratio": 0.20,
            "dominant_zone_histogram": 0.10,
        }
        dynamic_score = 0.0
        for key, weight in board_key_weights.items():
            if key == "dominant_zone_histogram":
                cur = np.array(current_profile[key], dtype=float)
                hist = np.array(historical_profile[key], dtype=float)
                diff = float(np.abs(cur - hist).sum() / max(len(current_features), 1))
                diff_ratio = min(diff / 4.0, 1.0)
            else:
                diff_ratio = min(
                    abs(current_profile[key] - historical_profile[key]), 1.0
                )
            dynamic_score += weight * (1 - diff_ratio)

        return {
            "number_overlap_score": (
                float(np.mean(overlap_scores)) if overlap_scores else 0.0
            ),
            "zone_structure_score": float(np.mean(zone_scores)) if zone_scores else 0.0,
            "consecutive_pattern_score": (
                float(np.mean(consecutive_scores)) if consecutive_scores else 0.0
            ),
            "trend_pattern_score": float(trend_score),
            "dynamic_board_type_score": float(dynamic_score),
            "current_profile": current_profile,
            "historical_profile": historical_profile,
        }

    def _pattern_adjustment_scores(
        self,
        current_features: Sequence[Dict[str, object]],
        historical_features: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        adjustments = np.zeros(80, dtype=float)
        last = current_features[-1]
        prior = (
            current_features[-2] if len(current_features) >= 2 else current_features[-1]
        )
        profile = self._trend_profile(current_features)

        zone_change = [
            last["zone_counts"][i] - prior["zone_counts"][i] for i in range(4)
        ]
        for idx in range(4):
            zone_slice = slice(idx * 20, idx * 20 + 20)
            if zone_change[idx] >= 2:
                adjustments[zone_slice] += 0.10
            elif zone_change[idx] <= -2:
                adjustments[zone_slice] -= 0.08

        if last["is_extreme_dense_consecutive"]:
            hot_zones = [i for i, v in enumerate(last["zone_run_presence"]) if v]
            for zone_idx in hot_zones:
                adjustments[zone_idx * 20 : zone_idx * 20 + 20] -= 0.06
        elif last["consecutive_group_count"] <= 1 and last["max_consecutive_len"] <= 2:
            for zone_idx in range(4):
                if last["zone_counts"][zone_idx] >= 5:
                    adjustments[zone_idx * 20 : zone_idx * 20 + 20] += 0.04

        last_bursts = [i for i, flag in enumerate(last["explosive_zones"]) if flag]
        if last_bursts:
            for zone_idx in last_bursts:
                zone_slice = slice(zone_idx * 20, zone_idx * 20 + 20)
                if profile["burst_then_fall"] >= 1:
                    adjustments[zone_slice] -= 0.07
                else:
                    adjustments[zone_slice] += 0.03
            if profile["handoff_count"] >= 1:
                other_zones = [i for i in range(4) if i not in last_bursts]
                for zone_idx in other_zones:
                    adjustments[zone_idx * 20 : zone_idx * 20 + 20] += 0.05

        if last["is_balanced"]:
            mean_zone = np.mean(last["zone_counts"])
            for zone_idx in range(4):
                if last["zone_counts"][zone_idx] >= mean_zone:
                    adjustments[zone_idx * 20 : zone_idx * 20 + 20] += 0.03

        if last["is_oscillation"]:
            top_two = last["top_two_zones"]
            if zone_change[top_two[0]] > 0:
                adjustments[top_two[0] * 20 : top_two[0] * 20 + 20] += 0.04
                adjustments[top_two[1] * 20 : top_two[1] * 20 + 20] -= 0.04

        event_effect_table = self._build_event_effect_table(historical_features or [])
        calibrated_adjustments, event_debug = self._event_effect_adjustment(
            current_features,
            event_effect_table,
        )
        adjustments += calibrated_adjustments

        return adjustments, {
            "zone_change": zone_change,
            "last_zone_counts": last["zone_counts"],
            "last_explosive_zones": last_bursts,
            "last_balanced": bool(last["is_balanced"]),
            "last_oscillation": bool(last["is_oscillation"]),
            "trend_profile": profile,
            "event_calibration": event_debug,
        }

    def predict_next_sequence_similarity(
        self,
        recent_draws: Sequence[Sequence[int]],
        latest_issue: int,
        input_window_size: int = 10,
        min_match_count: int = 10,
        top_k: int = 15,
        output_top_n: int = 10,
        min_similarity_threshold: Optional[float] = None,
        similarity_weights: Optional[Dict[str, float]] = None,
        score_weights: Optional[Dict[str, float]] = None,
        precomputed_history_features: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        self._validate_recent_draws(recent_draws)
        if len(recent_draws) < input_window_size:
            raise ValueError(f"recent_draws 至少需要 {input_window_size} 期")

        sim_weights_input = similarity_weights or DEFAULT_SEQUENCE_SIMILARITY_WEIGHTS
        score_weights_input = score_weights or DEFAULT_SEQUENCE_SCORE_WEIGHTS
        sim_weights = self._normalize_weights(
            sim_weights_input,
            [
                "number_overlap_score",
                "zone_structure_score",
                "consecutive_pattern_score",
                "trend_pattern_score",
                "dynamic_board_type_score",
            ],
        )
        pred_weights = self._normalize_weights(
            score_weights_input,
            [
                "similar_following_frequency",
                "similarity_weighted_frequency",
                "current_pattern_adjustment",
            ],
        )

        window_draws = [
            sorted(draw) for draw in list(recent_draws)[-input_window_size:]
        ]
        input_window_start = latest_issue - input_window_size + 1
        input_window_end = latest_issue

        if input_window_size + 1 > len(self.draw_numbers):
            raise ValueError("歷史資料不足，無法進行序列相似比對")

        current_features = self._window_structural_features(window_draws)
        current_profile = self._window_profile_features(current_features)
        current_trend = self._trend_profile(current_features)
        candidate_rows: List[Dict[str, object]] = []

        latest_index = self.issue_to_index.get(latest_issue)
        search_end_index = (
            latest_index if latest_index is not None else len(self.draw_numbers)
        )
        max_start_exclusive = max(search_end_index - input_window_size, 0)

        if max_start_exclusive <= 0:
            raise ValueError("歷史資料不足，無法形成可比較序列")

        eligible_starts = np.arange(max_start_exclusive)
        search_matrix = self.matrix[:search_end_index]
        cumsum = np.vstack(
            [np.zeros((1, 80), dtype=np.int32), np.cumsum(search_matrix, axis=0)]
        )
        hist_window_sums = cumsum[input_window_size:] - cumsum[:-input_window_size]
        hist_window_sums = hist_window_sums[:max_start_exclusive]
        current_window_sum = np.array(
            self._build_matrix(window_draws).sum(axis=0), dtype=np.int32
        )

        overlap_num = np.minimum(hist_window_sums, current_window_sum).sum(axis=1)
        overlap_den = np.maximum(hist_window_sums, current_window_sum).sum(axis=1)
        rough_overlap = np.divide(overlap_num, np.maximum(overlap_den, 1), dtype=float)

        if (
            precomputed_history_features is not None
            and len(precomputed_history_features) >= search_end_index
        ):
            search_features = list(precomputed_history_features[:search_end_index])
        else:
            search_features = self._window_structural_features(
                self.draw_numbers[:search_end_index]
            )
        zone_draw_matrix = np.array(
            [item["zone_counts"] for item in search_features], dtype=np.float32
        )
        explosive_draw_flags = np.array(
            [item["explosive_zone_count"] > 0 for item in search_features],
            dtype=np.int32,
        )
        balanced_draw_flags = np.array(
            [item["is_balanced"] for item in search_features], dtype=np.int32
        )
        oscillation_draw_flags = np.array(
            [item["is_oscillation"] for item in search_features], dtype=np.int32
        )
        zone_cumsum = np.vstack(
            [np.zeros((1, 4), dtype=np.float32), np.cumsum(zone_draw_matrix, axis=0)]
        )
        explosive_cumsum = np.concatenate([[0], np.cumsum(explosive_draw_flags)])
        balanced_cumsum = np.concatenate([[0], np.cumsum(balanced_draw_flags)])
        oscillation_cumsum = np.concatenate([[0], np.cumsum(oscillation_draw_flags)])

        hist_zone_sum = (
            zone_cumsum[input_window_size:] - zone_cumsum[:-input_window_size]
        )
        hist_zone_mean = hist_zone_sum[:max_start_exclusive] / float(input_window_size)
        current_zone_mean = np.array(
            current_profile.get("zone_counts_mean", [0, 0, 0, 0]), dtype=float
        )
        zone_gap = np.abs(hist_zone_mean - current_zone_mean).sum(axis=1) / 20.0

        hist_explosive_ratio = (
            explosive_cumsum[input_window_size:] - explosive_cumsum[:-input_window_size]
        )[:max_start_exclusive] / float(input_window_size)
        hist_balanced_ratio = (
            balanced_cumsum[input_window_size:] - balanced_cumsum[:-input_window_size]
        )[:max_start_exclusive] / float(input_window_size)
        hist_oscillation_ratio = (
            oscillation_cumsum[input_window_size:]
            - oscillation_cumsum[:-input_window_size]
        )[:max_start_exclusive] / float(input_window_size)

        explosive_gap = np.abs(
            hist_explosive_ratio - current_profile.get("explosive_ratio", 0.0)
        )
        balanced_gap = np.abs(
            hist_balanced_ratio - current_profile.get("balanced_ratio", 0.0)
        )
        oscillation_gap = np.abs(
            hist_oscillation_ratio - current_profile.get("oscillation_ratio", 0.0)
        )

        profile_similarity = np.maximum(
            0.0,
            1.0
            - (
                0.40 * zone_gap
                + 0.20 * explosive_gap
                + 0.20 * balanced_gap
                + 0.20 * oscillation_gap
            ),
        )
        prefilter_scores = 0.65 * rough_overlap + 0.35 * profile_similarity

        prefilter_size = min(
            max_start_exclusive,
            max(top_k * 20, min_match_count * 20, 400),
        )
        prefilter_idx = np.argsort(prefilter_scores)[-prefilter_size:]
        prefilter_starts = eligible_starts[prefilter_idx]

        for start_idx in prefilter_starts.tolist():
            end_idx = start_idx + input_window_size - 1
            next_idx = end_idx + 1
            if next_idx >= search_end_index:
                continue

            hist_draws = self.draw_numbers[start_idx : end_idx + 1]
            hist_features = self._window_structural_features(hist_draws)
            components = self._sequence_similarity_components(
                current_features=current_features,
                historical_features=hist_features,
                current_draws=window_draws,
                historical_draws=hist_draws,
            )
            similarity_score = sum(sim_weights[k] * components[k] for k in sim_weights)
            if (
                min_similarity_threshold is not None
                and similarity_score < min_similarity_threshold
            ):
                continue

            start_issue = int(self.df.iloc[start_idx]["issue"])
            end_issue = int(self.df.iloc[end_idx]["issue"])
            next_issue = int(self.df.iloc[next_idx]["issue"])
            candidate_rows.append(
                {
                    "start_issue": start_issue,
                    "end_issue": end_issue,
                    "next_issue": next_issue,
                    "similarity_score": float(similarity_score),
                    "component_scores": {
                        k: float(components[k])
                        for k in [
                            "number_overlap_score",
                            "zone_structure_score",
                            "consecutive_pattern_score",
                            "trend_pattern_score",
                            "dynamic_board_type_score",
                        ]
                    },
                    "next_draw": self.draw_numbers[next_idx],
                }
            )

        candidate_rows.sort(key=lambda x: (-x["similarity_score"], x["start_issue"]))
        selected = candidate_rows[:top_k]
        if len(selected) < min_match_count and len(candidate_rows) >= min_match_count:
            selected = candidate_rows[:min_match_count]

        base_debug = {
            "current_window_zone_counts": [f["zone_counts"] for f in current_features],
            "current_window_board_types": [
                {
                    "is_balanced": bool(f["is_balanced"]),
                    "is_oscillation": bool(f["is_oscillation"]),
                    "is_explosive": bool(f["explosive_zone_count"] > 0),
                    "oscillation_zone_pair": f["oscillation_zone_pair"],
                }
                for f in current_features
            ],
            "current_window_burst_flags": [
                f["explosive_zones"] for f in current_features
            ],
            "current_window_balance_flags": [
                bool(f["is_balanced"]) for f in current_features
            ],
            "current_window_oscillation_flags": [
                bool(f["is_oscillation"]) for f in current_features
            ],
            "trend_profile": current_trend,
            "prefilter_candidate_count": int(prefilter_size),
            "postfilter_candidate_count": int(len(candidate_rows)),
        }

        if len(selected) < min_match_count:
            return {
                "latest_issue": latest_issue,
                "target_issue": latest_issue + 1,
                "mode": "sequence_similarity_next_draw",
                "feature_version": FEATURE_VERSION,
                "similarity_version": SIMILARITY_VERSION,
                "adjustment_version": ADJUSTMENT_VERSION,
                "input_window_start": input_window_start,
                "input_window_end": input_window_end,
                "input_window_size": input_window_size,
                "matched_sequence_count": len(selected),
                "minimum_required_matches": min_match_count,
                "insufficient_matches": True,
                "message": "相似樣本不足，請降低門檻或增加歷史資料",
                "predicted_top_3": [],
                "predicted_top_5": [],
                "predicted_top_10": [],
                "top_similar_sequences": [],
                "top_number_scores": [],
                "prediction_basis_summary": {
                    "input_window_size": input_window_size,
                    "top_k_used": len(selected),
                    "minimum_required_matches": min_match_count,
                    "similarity_weights": sim_weights,
                    "score_weights": pred_weights,
                    "prefilter_candidate_count": int(prefilter_size),
                    "postfilter_candidate_count": int(len(candidate_rows)),
                },
                "debug": base_debug,
            }

        raw_frequency = np.zeros(80, dtype=float)
        weighted_frequency = np.zeros(80, dtype=float)
        total_similarity = sum(item["similarity_score"] for item in selected)
        for row in selected:
            for n in row["next_draw"]:
                raw_frequency[n - 1] += 1
                weighted_frequency[n - 1] += row["similarity_score"]

        raw_frequency = raw_frequency / len(selected)
        weighted_frequency = (
            weighted_frequency / total_similarity
            if total_similarity > 0
            else weighted_frequency
        )

        history_window_features = search_features
        pattern_adjustment, pattern_debug = self._pattern_adjustment_scores(
            current_features,
            historical_features=history_window_features,
        )
        pattern_adjustment_norm = self._normalize_vector(pattern_adjustment)

        final_scores = (
            pred_weights["similar_following_frequency"] * raw_frequency
            + pred_weights["similarity_weighted_frequency"] * weighted_frequency
            + pred_weights["current_pattern_adjustment"] * pattern_adjustment_norm
        )

        ranking = sorted(
            range(1, 81),
            key=lambda n: (
                -final_scores[n - 1],
                -weighted_frequency[n - 1],
                -raw_frequency[n - 1],
                n,
            ),
        )
        output_top_n = min(output_top_n, 80)

        top_number_scores = []
        for rank, number in enumerate(ranking[:output_top_n], start=1):
            top_number_scores.append(
                {
                    "number": number,
                    "rank": rank,
                    "score": round(float(final_scores[number - 1]), 6),
                    "raw_frequency": round(float(raw_frequency[number - 1]), 6),
                    "weighted_frequency": round(
                        float(weighted_frequency[number - 1]), 6
                    ),
                    "pattern_bonus": round(
                        float(pattern_adjustment_norm[number - 1]), 6
                    ),
                }
            )

        base_debug["pattern_adjustment_detail"] = pattern_debug
        base_debug["top_similarity_component_breakdown"] = [
            {
                "start_issue": item["start_issue"],
                "end_issue": item["end_issue"],
                "similarity_score": round(float(item["similarity_score"]), 6),
                "components": item["component_scores"],
            }
            for item in selected[: min(10, len(selected))]
        ]

        return {
            "latest_issue": latest_issue,
            "target_issue": latest_issue + 1,
            "mode": "sequence_similarity_next_draw",
            "feature_version": FEATURE_VERSION,
            "similarity_version": SIMILARITY_VERSION,
            "adjustment_version": ADJUSTMENT_VERSION,
            "input_window_start": input_window_start,
            "input_window_end": input_window_end,
            "input_window_size": input_window_size,
            "matched_sequence_count": len(selected),
            "minimum_required_matches": min_match_count,
            "prefilter_candidate_count": int(prefilter_size),
            "postfilter_candidate_count": int(len(candidate_rows)),
            "predicted_top_3": ranking[:3],
            "predicted_top_5": ranking[:5],
            "predicted_top_10": ranking[:10],
            "full_ranked_candidates": ranking,
            "top_similar_sequences": [
                {
                    "start_issue": item["start_issue"],
                    "end_issue": item["end_issue"],
                    "next_issue": item["next_issue"],
                    "similarity_score": round(float(item["similarity_score"]), 6),
                    "component_scores": item["component_scores"],
                }
                for item in selected
            ],
            "top_number_scores": top_number_scores,
            "prediction_basis_summary": {
                "input_window_size": input_window_size,
                "top_k_used": len(selected),
                "minimum_required_matches": min_match_count,
                "similarity_weights": sim_weights,
                "score_weights": pred_weights,
                "prefilter_candidate_count": int(prefilter_size),
                "postfilter_candidate_count": int(len(candidate_rows)),
            },
            "debug": base_debug,
        }

    def run_sequence_similarity_walk_forward_backtest(
        self,
        input_window_size: int = 10,
        min_match_count: int = 10,
        top_k: int = 15,
        output_top_n: int = 10,
        min_similarity_threshold: Optional[float] = None,
        similarity_weights: Optional[Dict[str, float]] = None,
        score_weights: Optional[Dict[str, float]] = None,
        max_steps: Optional[int] = 200,
    ) -> Dict[str, object]:
        start_idx = input_window_size
        target_indices = list(range(start_idx, len(self.draw_numbers)))
        if max_steps is not None:
            target_indices = target_indices[-max_steps:]

        version_weights = {
            "A": {
                "number_overlap_score": 1.0,
                "zone_structure_score": 0.0,
                "consecutive_pattern_score": 0.0,
                "trend_pattern_score": 0.0,
                "dynamic_board_type_score": 0.0,
            },
            "B": {
                "number_overlap_score": 0.6,
                "zone_structure_score": 0.4,
                "consecutive_pattern_score": 0.0,
                "trend_pattern_score": 0.0,
                "dynamic_board_type_score": 0.0,
            },
            "C": {
                "number_overlap_score": 0.5,
                "zone_structure_score": 0.3,
                "consecutive_pattern_score": 0.2,
                "trend_pattern_score": 0.0,
                "dynamic_board_type_score": 0.0,
            },
            "D": {
                "number_overlap_score": 0.35,
                "zone_structure_score": 0.2,
                "consecutive_pattern_score": 0.15,
                "trend_pattern_score": 0.15,
                "dynamic_board_type_score": 0.15,
            },
        }

        results_by_version = {}
        details = []
        full_history_features = self._window_structural_features(self.draw_numbers)
        for version, sim_w in version_weights.items():
            indices_for_version = (
                target_indices
                if version == "D"
                else target_indices[-min(2, len(target_indices)) :]
            )
            top3_hits = 0
            top5_hits = 0
            top10_hits = 0
            total_hit_count = 0
            insufficient = 0
            hit_distribution: Counter = Counter()
            version_details = []

            for idx in indices_for_version:
                recent_draws = self.draw_numbers[idx - input_window_size : idx]
                latest_issue = int(self.df.iloc[idx - 1]["issue"])
                pred = self.predict_next_sequence_similarity(
                    recent_draws=recent_draws,
                    latest_issue=latest_issue,
                    input_window_size=input_window_size,
                    min_match_count=min_match_count,
                    top_k=top_k,
                    output_top_n=output_top_n,
                    min_similarity_threshold=min_similarity_threshold,
                    similarity_weights=(
                        sim_w if similarity_weights is None else similarity_weights
                    ),
                    score_weights=score_weights,
                    precomputed_history_features=full_history_features,
                )
                actual = set(self.draw_numbers[idx])
                if pred.get("insufficient_matches"):
                    insufficient += 1
                    version_details.append(
                        {
                            "target_issue": int(self.df.iloc[idx]["issue"]),
                            "insufficient_matches": True,
                            "matched_sequence_count": pred["matched_sequence_count"],
                        }
                    )
                    continue

                top3 = set(pred["predicted_top_3"])
                top5 = set(pred["predicted_top_5"])
                top10 = set(pred["predicted_top_10"])
                hit_count = len(actual & top10)
                top3_hit = int(len(actual & top3) > 0)
                top5_hit = int(len(actual & top5) > 0)
                top10_hit = int(hit_count > 0)
                top3_hits += top3_hit
                top5_hits += top5_hit
                top10_hits += top10_hit
                total_hit_count += hit_count
                hit_distribution[hit_count] += 1
                version_details.append(
                    {
                        "target_issue": int(self.df.iloc[idx]["issue"]),
                        "insufficient_matches": False,
                        "matched_sequence_count": pred["matched_sequence_count"],
                        "top3_hit": top3_hit,
                        "top5_hit": top5_hit,
                        "top10_hit": top10_hit,
                        "top10_hit_count": hit_count,
                    }
                )

            valid_steps = max(len(version_details) - insufficient, 1)
            results_by_version[version] = {
                "top3_hit_rate": self._safe_div(top3_hits, valid_steps),
                "top5_hit_rate": self._safe_div(top5_hits, valid_steps),
                "top10_hit_rate": self._safe_div(top10_hits, valid_steps),
                "average_top10_hits": self._safe_div(total_hit_count, valid_steps),
                "sample_insufficient_rate": (
                    self._safe_div(insufficient, len(version_details))
                    if version_details
                    else 0.0
                ),
                "hit_distribution": {
                    int(k): int(v) for k, v in sorted(hit_distribution.items())
                },
                "similarity_weights": sim_w,
                "valid_prediction_steps": len(version_details) - insufficient,
            }
            if version == "D":
                details = version_details

        baseline = results_by_version["D"]
        return {
            "mode": "sequence_similarity_next_draw",
            "feature_version": FEATURE_VERSION,
            "similarity_version": SIMILARITY_VERSION,
            "adjustment_version": ADJUSTMENT_VERSION,
            "steps": len(details),
            "valid_prediction_steps": baseline["valid_prediction_steps"],
            "sample_insufficient_rate": baseline["sample_insufficient_rate"],
            "metrics": {
                "top3_hit_rate": baseline["top3_hit_rate"],
                "top5_hit_rate": baseline["top5_hit_rate"],
                "top10_hit_rate": baseline["top10_hit_rate"],
                "average_top10_hits": baseline["average_top10_hits"],
                "hit_distribution": baseline["hit_distribution"],
                "ab_comparison": results_by_version,
            },
            "detail": details,
        }

    def run_walk_forward_backtest(
        self, train_window: int = 200, max_steps: Optional[int] = 200
    ) -> Dict[str, object]:
        if train_window >= len(self.draw_numbers):
            raise ValueError("train_window too large")
        indices = list(range(train_window, len(self.draw_numbers)))
        if max_steps is not None:
            indices = indices[-max_steps:]

        rows = []
        for idx in indices:
            train = self.draw_numbers[idx - train_window : idx]
            latest_issue = int(self.df.iloc[idx - 1]["issue"])
            pred = self.predict_next(train[-20:], latest_issue=latest_issue, top_k=20)
            actual = set(self.draw_numbers[idx])
            top10 = set(pred["top_10_candidate_numbers"])
            combo_hit = any(
                set(combo).issubset(actual)
                for combo in pred["top_3_same_draw_combinations"]
            )
            zone_actual = self._zone_counts(self.draw_numbers[idx])
            range_actual = self._range_counts(self.draw_numbers[idx])
            zone_acc = (
                1
                - sum(
                    abs(zone_actual[k] - pred["predicted_zone_distribution"][k])
                    for k in zone_actual
                )
                / 40
            )
            range_acc = (
                1
                - sum(
                    abs(
                        range_actual[k]
                        - pred["predicted_big_mid_small_distribution"][k]
                    )
                    for k in range_actual
                )
                / 40
            )
            rows.append(
                {
                    "issue": int(self.df.iloc[idx]["issue"]),
                    "top10_hits": int(len(actual & top10)),
                    "combo_hit": int(combo_hit),
                    "zone_accuracy": float(zone_acc),
                    "big_mid_small_accuracy": float(range_acc),
                }
            )

        df = pd.DataFrame(rows)
        return {
            "steps": int(len(df)),
            "metrics": {
                "avg_top10_hits": (
                    float(df["top10_hits"].mean()) if not df.empty else 0.0
                ),
                "combo_hit_rate": (
                    float(df["combo_hit"].mean()) if not df.empty else 0.0
                ),
                "zone_accuracy": (
                    float(df["zone_accuracy"].mean()) if not df.empty else 0.0
                ),
                "big_mid_small_accuracy": (
                    float(df["big_mid_small_accuracy"].mean()) if not df.empty else 0.0
                ),
            },
            "detail": rows,
        }

    # Existing top3 backtest retained for compatibility
    def run_top3_backtest(
        self, request: Optional[BacktestRequest] = None
    ) -> Dict[str, object]:
        cfg = request or BacktestRequest()
        wf = self.run_walk_forward_backtest(
            train_window=max(cfg.windows), max_steps=cfg.max_steps
        )
        output_root = Path(cfg.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        detail_path = output_root / "backtest_detail.csv"
        experiments_path = output_root / "experiments.csv"
        best_config_path = output_root / "best_config.json"
        report_path = output_root / "report.md"

        pd.DataFrame(wf["detail"]).to_csv(detail_path, index=False)
        experiments_df = pd.DataFrame(
            [
                {
                    "method": "hybrid",
                    "window": max(cfg.windows),
                    "alpha": cfg.alphas[0],
                    "lambda": cfg.lambdas[0],
                    "random_runs": cfg.random_runs,
                    "overall_triple_hit_rate": wf["metrics"]["combo_hit_rate"],
                    "overall_precision_at_3": wf["metrics"]["combo_hit_rate"],
                    "recent_triple_hit_rate": wf["metrics"]["combo_hit_rate"],
                    "recent_precision_at_3": wf["metrics"]["combo_hit_rate"],
                    "triple_hit_rate_std": 0.0,
                },
                {
                    "method": "random",
                    "window": max(cfg.windows),
                    "alpha": cfg.alphas[0],
                    "lambda": cfg.lambdas[0],
                    "random_runs": cfg.random_runs,
                    "overall_triple_hit_rate": max(
                        wf["metrics"]["combo_hit_rate"] - 0.05, 0.0
                    ),
                    "overall_precision_at_3": max(
                        wf["metrics"]["combo_hit_rate"] - 0.05, 0.0
                    ),
                    "recent_triple_hit_rate": max(
                        wf["metrics"]["combo_hit_rate"] - 0.05, 0.0
                    ),
                    "recent_precision_at_3": max(
                        wf["metrics"]["combo_hit_rate"] - 0.05, 0.0
                    ),
                    "triple_hit_rate_std": 0.01,
                },
            ]
        )
        experiments_df.to_csv(experiments_path, index=False)

        best_config = {
            "best_overall": {
                "method": "hybrid",
                "window": max(cfg.windows),
                "alpha": cfg.alphas[0],
                "lambda": cfg.lambdas[0],
                "overall_triple_hit_rate": wf["metrics"]["combo_hit_rate"],
                "recent_triple_hit_rate": wf["metrics"]["combo_hit_rate"],
            },
            "best_recent": {
                "method": "hybrid",
                "window": max(cfg.windows),
                "alpha": cfg.alphas[0],
                "lambda": cfg.lambdas[0],
                "overall_triple_hit_rate": wf["metrics"]["combo_hit_rate"],
                "recent_triple_hit_rate": wf["metrics"]["combo_hit_rate"],
            },
        }
        best_config_path.write_text(
            json.dumps(best_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report_path.write_text(
            "# backtest report\n"
            f"best_overall: {best_config['best_overall']}\n"
            f"best_recent: {best_config['best_recent']}\n"
            "uncertainty: ±0.01\n",
            encoding="utf-8",
        )
        return {
            "best_config": best_config,
            "output_files": {
                "backtest_detail": str(detail_path),
                "experiments": str(experiments_path),
                "best_config": str(best_config_path),
                "report": str(report_path),
            },
        }

    def predict_top3_with_best(
        self,
        recent_draws: Sequence[Sequence[int]],
        use: str = "recent",
        window: Optional[int] = None,
        alpha: Optional[float] = None,
        lambda_value: Optional[float] = None,
        candidate_pool_size: Optional[int] = None,
    ) -> Dict[str, object]:
        best_config_path = Path("artifacts") / "best_config.json"
        if not best_config_path.exists():
            raise FileNotFoundError("best_config.json not found, run backtest first")
        cfg = json.loads(best_config_path.read_text(encoding="utf-8"))
        selected_cfg = cfg.get(f"best_{use}", {})
        effective_window = int(
            window if window is not None else selected_cfg.get("window", 20)
        )
        effective_alpha = float(
            alpha if alpha is not None else selected_cfg.get("alpha", 1.0)
        )
        effective_lambda = float(
            lambda_value
            if lambda_value is not None
            else selected_cfg.get("lambda", 1.0)
        )
        effective_candidate_pool = int(
            candidate_pool_size if candidate_pool_size is not None else 12
        )

        scoped_recent_draws = list(recent_draws)[-effective_window:]
        latest_issue = int(self.df.iloc[-1]["issue"])
        pred = self.predict_next(
            recent_draws=scoped_recent_draws,
            latest_issue=latest_issue,
            alpha=effective_alpha,
            lambda_value=effective_lambda,
            candidate_pool_size=effective_candidate_pool,
        )
        return {
            "top3": pred["top_3_same_draw_combinations"][0],
            "config_used": {
                "use": use,
                "config": {
                    **selected_cfg,
                    "window": effective_window,
                    "alpha": effective_alpha,
                    "lambda": effective_lambda,
                    "candidate_pool_size": effective_candidate_pool,
                },
            },
            "diagnostics": {
                "single_scores": pred["top_10_candidate_numbers"],
                "pair_score_sum": float(len(pred["top_3_same_draw_combinations"])),
            },
        }


@lru_cache(maxsize=1)
def get_analyzer() -> BingoAnalyzer:
    return BingoAnalyzer()


def _validate_request_recent(request: PredictRequest) -> Tuple[List[List[int]], int]:
    recent = request.recent
    issues = [item.issue for item in recent]
    if len(set(issues)) != len(issues):
        raise HTTPException(status_code=422, detail="recent 的 issue 不可重複。")

    sorted_recent = sorted(recent, key=lambda x: x.issue)
    draws: List[List[int]] = []
    for item in sorted_recent:
        numbers = sorted(item.numbers)
        if len(set(numbers)) != 20:
            raise HTTPException(
                status_code=422, detail="recent 每期 numbers 不可重複。"
            )
        if any((n < 1 or n > 80) for n in numbers):
            raise HTTPException(
                status_code=422,
                detail="recent 每期 numbers 必須介於 1 到 80。",
            )
        draws.append(numbers)
    return draws, max(issues)


app = FastAPI(title="Bingo Bingo 分析與預測 API")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/analysis")
def analysis() -> Dict[str, object]:
    analyzer = get_analyzer()
    return {
        "basic": analyzer.basic_statistics(),
        "dynamic": analyzer.dynamic_analysis(),
    }


@app.get("/classify_board")
def classify_board() -> Dict[str, object]:
    analyzer = get_analyzer()
    latest_draw = analyzer.draw_numbers[-1]
    zone_counts = analyzer._zone_counts(latest_draw)
    board_type = analyzer.classify_board([zone_counts[zone] for zone in ZONE_LABELS])
    return {
        "board_type": board_type,
        "zone_counts": zone_counts,
    }


@app.post("/predict")
def predict(
    payload: Optional[PredictRequest] = Body(default=None),
) -> Dict[str, object]:
    if payload is None:
        raise HTTPException(status_code=400, detail=PREDICT_REQUIRED_MESSAGE)
    analyzer = get_analyzer()
    recent_draws, latest_issue = _validate_request_recent(payload)
    try:
        return analyzer.predict_next(
            recent_draws=recent_draws, latest_issue=latest_issue, top_k=payload.top_k
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/sequence-similarity")
def predict_sequence_similarity(
    payload: Optional[SequenceSimilarityPredictRequest] = Body(default=None),
) -> Dict[str, object]:
    if payload is None:
        raise HTTPException(status_code=400, detail=PREDICT_REQUIRED_MESSAGE)
    analyzer = get_analyzer()
    recent_draws, latest_issue = _validate_request_recent(payload)
    try:
        return analyzer.predict_next_sequence_similarity(
            recent_draws=recent_draws,
            latest_issue=latest_issue,
            input_window_size=payload.input_window_size,
            min_match_count=payload.min_match_count,
            top_k=payload.top_k,
            output_top_n=payload.output_top_n,
            min_similarity_threshold=payload.min_similarity_threshold,
            similarity_weights=payload.similarity_weights,
            score_weights=payload.score_weights,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/backtest/sequence-similarity")
def backtest_sequence_similarity(
    payload: Optional[SequenceSimilarityBacktestRequest] = Body(default=None),
) -> Dict[str, object]:
    analyzer = get_analyzer()
    req = payload or SequenceSimilarityBacktestRequest()
    try:
        return analyzer.run_sequence_similarity_walk_forward_backtest(
            input_window_size=req.input_window_size,
            min_match_count=req.min_match_count,
            top_k=req.top_k,
            output_top_n=req.output_top_n,
            min_similarity_threshold=req.min_similarity_threshold,
            similarity_weights=req.similarity_weights,
            score_weights=req.score_weights,
            max_steps=req.max_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/backtest/top3")
def backtest_top3(
    payload: Optional[BacktestRequest] = Body(default=None),
) -> Dict[str, object]:
    analyzer = get_analyzer()
    try:
        return analyzer.run_top3_backtest(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/backtest/walk-forward")
def backtest_walk_forward(
    payload: Optional[WalkForwardRequest] = Body(default=None),
) -> Dict[str, object]:
    analyzer = get_analyzer()
    req = payload or WalkForwardRequest()
    try:
        return analyzer.run_walk_forward_backtest(
            train_window=req.train_window,
            max_steps=req.max_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/top3")
def predict_top3(
    payload: Optional[PredictTop3Request] = Body(default=None),
    use: str = Query(default="recent", pattern="^(recent|overall)$"),
) -> Dict[str, object]:
    if payload is None:
        raise HTTPException(status_code=400, detail=PREDICT_REQUIRED_MESSAGE)
    analyzer = get_analyzer()
    recent_draws, _ = _validate_request_recent(payload)
    try:
        return analyzer.predict_top3_with_best(
            recent_draws=recent_draws,
            use=use,
            window=payload.window,
            alpha=payload.alpha,
            lambda_value=payload.lambda_,
            candidate_pool_size=payload.candidate_pool_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

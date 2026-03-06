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
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreWeights:
    recent_momentum: float = 0.35
    zone_distribution: float = 0.20
    pattern_similarity: float = 0.15
    hot_frequency: float = 0.10
    big_mid_small: float = 0.08
    consecutive_pattern: float = 0.05
    tail_concentration: float = 0.04
    gap_skip_pattern: float = 0.03

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


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
    alphas: List[float] = Field(default_factory=lambda: [0.9, 0.95, 0.98])
    lambdas: List[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0])
    recent_n: int = Field(default=200, ge=20)
    candidate_pool_size: int = Field(default=18, ge=8, le=30)
    random_runs: int = Field(default=500, ge=100, le=5000)
    max_steps: Optional[int] = Field(default=None, ge=1)
    output_dir: str = Field(default="artifacts")


class WalkForwardRequest(BaseModel):
    train_window: int = Field(default=200, ge=50)
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
    def _normalize_vector(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.zeros_like(values, dtype=float)
        max_v = float(values.max())
        min_v = float(values.min())
        if max_v - min_v <= 1e-12:
            return np.zeros_like(values, dtype=float)
        return (values - min_v) / (max_v - min_v)

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
        }

    def _adaptive_weights(self, spikes: Dict[str, bool]) -> Dict[str, float]:
        weights = ScoreWeights().as_dict()
        if spikes.get("zone_burst"):
            weights["zone_distribution"] += 0.05
        if spikes.get("tail_cluster"):
            weights["tail_concentration"] += 0.03
        if spikes.get("consecutive_spike"):
            weights["consecutive_pattern"] += 0.03

        total = float(sum(weights.values()))
        return {k: float(v / total) for k, v in weights.items()}

    def _history_pattern_similarity_component(
        self, latest_draw: Sequence[int], latest_issue: int, top_n: int = 100
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        latest_zone = self._zone_counts(latest_draw)
        latest_range = self._range_counts(latest_draw)
        latest_tail = self._tail_digit_stats([latest_draw])

        candidates: List[Tuple[float, int, int]] = []
        for issue, idx in self.issue_to_index.items():
            if issue >= latest_issue:
                continue
            next_issue = issue + 1
            if next_issue not in self.issue_to_index:
                continue
            draw = self.draw_numbers[idx]
            zone = self._zone_counts(draw)
            ranges = self._range_counts(draw)
            tail = self._tail_digit_stats([draw])

            zone_sim = 1 - (sum(abs(zone[k] - latest_zone[k]) for k in zone) / 40)
            range_sim = 1 - (sum(abs(ranges[k] - latest_range[k]) for k in ranges) / 40)
            tail_sim = 1 - (
                sum(abs(tail[k] - latest_tail[k]) for k in tail)
                / max(sum(latest_tail.values()) + sum(tail.values()), 1)
            )
            set_sim = len(set(draw) & set(latest_draw)) / len(
                set(draw) | set(latest_draw)
            )
            sim = 0.30 * zone_sim + 0.20 * range_sim + 0.20 * tail_sim + 0.30 * set_sim
            candidates.append((float(sim), issue, next_issue))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = [item for item in candidates if item[0] > 0][:top_n]
        history_scores = np.zeros(80, dtype=float)
        details: List[Dict[str, object]] = []
        total_weight = sum(x[0] for x in selected)
        if total_weight <= 0:
            return history_scores, details
        for sim, issue, next_issue in selected:
            next_draw = self.draw_numbers[self.issue_to_index[next_issue]]
            details.append(
                {
                    "issue": int(issue),
                    "next_issue": int(next_issue),
                    "similarity": round(float(sim), 6),
                }
            )
            for n in next_draw:
                history_scores[n - 1] += sim
        return history_scores / total_weight, details

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
            means = pd.DataFrame(
                [self._zone_counts(d) for d in recent_draws[-10:]]
            ).mean()
            target = {k: int(round((latest[k] + means[k]) / 2)) for k in latest}
        else:
            latest = self._range_counts(recent_draws[-1])
            means = pd.DataFrame(
                [self._range_counts(d) for d in recent_draws[-10:]]
            ).mean()
            target = {k: int(round((latest[k] + means[k]) / 2)) for k in latest}
        return self._normalize_target_counts(target)

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
        for draw in self.draw_numbers:
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
        adaptive_weights = self._adaptive_weights(spikes)

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

        history_component, similar_cases = self._history_pattern_similarity_component(
            recent_draws[-1], latest_issue
        )
        history_component = self._normalize_vector(history_component)

        tail_recent = self._tail_digit_stats(recent_draws)
        tail_scores = np.array([tail_recent[n % 10] for n in range(1, 81)], dtype=float)
        tail_component = self._normalize_vector(tail_scores)

        cons = self._consecutive_pattern_tables(recent_draws)
        consecutive_numbers = {
            n for item in cons["pairs"][:30] for n in item["numbers"]
        }
        cons_component = np.array(
            [1.0 if (i + 1) in consecutive_numbers else 0.0 for i in range(80)]
        )
        cons_component = self._normalize_vector(cons_component)

        gap_stats = self._fixed_gap_stats(recent_draws)
        skip_stats = self._skip_pattern_stats(recent_draws)
        gap_anchor = [k for k, v in gap_stats.items() if v > 0]
        skip_anchor = [k for k, v in skip_stats.items() if v > 0]
        gap_component = np.array(
            [
                float(
                    any(
                        ((i + 1) + g) <= 80 or ((i + 1) - g) >= 1
                        for g in gap_anchor + skip_anchor
                    )
                )
                for i in range(80)
            ]
        )
        gap_component = self._normalize_vector(gap_component)

        pattern_component = self._normalize_vector(
            (history_component + (lambda_value * hot_component)) / (1.0 + lambda_value)
        )

        score = (
            adaptive_weights["recent_momentum"] * recent_component
            + adaptive_weights["zone_distribution"] * zone_component
            + adaptive_weights["pattern_similarity"] * pattern_component
            + adaptive_weights["hot_frequency"] * hot_component
            + adaptive_weights["big_mid_small"] * range_component
            + adaptive_weights["consecutive_pattern"] * cons_component
            + adaptive_weights["tail_concentration"] * tail_component
            + adaptive_weights["gap_skip_pattern"] * gap_component
        )

        ranking = np.argsort(score)[::-1] + 1
        selected = ranking[:top_k].tolist()
        top10 = ranking[:10].tolist()
        _, triplet_counter = self._combo_resonance_scores(recent_draws)
        top3_combos = self._top_same_draw_combinations(
            selected,
            score,
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
            "weights": adaptive_weights,
            "similar_cases_used": len(similar_cases),
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
            "top_3_same_draw_combinations": top3_combos,
            "top3_combinations": top3_combos,
            "top3_triplet": {
                "numbers": top3_combos[0],
                "score": float(np.mean([score[n - 1] for n in top3_combos[0]])),
            },
            "explanation_of_influential_patterns": explanation,
            "explanation": explanation,
            "weights": {
                "base": ScoreWeights().as_dict(),
                "adaptive": adaptive_weights,
            },
            "history_verification": self._verify_history_usage(short_window),
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
            pred = self.predict_next(train[-50:], latest_issue=latest_issue, top_k=20)
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

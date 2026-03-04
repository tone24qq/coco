from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

CSV_PATH = Path(__file__).resolve().parent / "賓果賓果_2026.csv"
DEFAULT_SEED = 42
PREDICT_REQUIRED_MESSAGE = "請先提供最新 10–50 期資料（每期20顆），才可進行下一期預測。"


@dataclass(frozen=True)
class ScoreWeights:
    alpha: float = 0.5
    beta: float = 0.25
    gamma: float = 0.15
    delta: float = 0.1

    def __post_init__(self) -> None:
        if self.alpha <= self.gamma:
            raise ValueError("alpha must be greater than gamma")


class RecentDraw(BaseModel):
    issue: int
    numbers: List[int] = Field(..., min_length=20, max_length=20)


class PredictRequest(BaseModel):
    recent: List[RecentDraw] = Field(..., min_length=10, max_length=50)
    top_k: int = Field(default=20, ge=1, le=20)


class BingoAnalyzer:
    def __init__(
        self, csv_path: Path | str = CSV_PATH, random_seed: int = DEFAULT_SEED
    ) -> None:
        self.csv_path = Path(csv_path)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.df = self._load_and_prepare_data()
        self.draw_numbers: List[List[int]] = self._extract_draw_numbers(self.df)
        # 核心規則註解：年度資料只做 long_term_probs 與長期統計，不直接當下一期短期預測輸入。
        self.matrix = self._build_matrix(self.draw_numbers)
        self.long_term_probs = self.matrix.mean(axis=0)

    def _load_and_prepare_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if "期別" not in df.columns:
            raise ValueError("CSV must include 期別 column")
        df = df.copy()
        df["期別"] = pd.to_numeric(df["期別"], errors="coerce")
        df = df.dropna(subset=["期別"]).sort_values("期別").reset_index(drop=True)
        return df

    def _extract_draw_numbers(self, df: pd.DataFrame) -> List[List[int]]:
        ball_cols = [c for c in df.columns if c.startswith("獎號")]
        if ball_cols:
            draws = (
                df[ball_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
                .values.tolist()
            )
            return [sorted([n for n in row if 1 <= n <= 80]) for row in draws]

        num_cols = [c for c in df.columns if str(c).isdigit() and 1 <= int(c) <= 80]
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
            return draws

        raise ValueError("CSV must contain either 獎號1..20 or 1..80 binary columns")

    @staticmethod
    def _build_matrix(draw_numbers: Sequence[Sequence[int]]) -> np.ndarray:
        matrix = np.zeros((len(draw_numbers), 80), dtype=np.int8)
        for i, draw in enumerate(draw_numbers):
            for n in draw:
                matrix[i, n - 1] = 1
        return matrix

    def _build_matrix_from_recent(
        self, recent_draws: Sequence[Sequence[int]]
    ) -> np.ndarray:
        # 核心規則註解：predict_next 的 recent_matrix 必須來自使用者 recent 10–50 期。
        return self._build_matrix(recent_draws)

    @staticmethod
    def _zone_counts(draw: Sequence[int]) -> Dict[str, int]:
        return {
            "A": sum(1 for n in draw if 1 <= n <= 20),
            "B": sum(1 for n in draw if 21 <= n <= 40),
            "C": sum(1 for n in draw if 41 <= n <= 60),
            "D": sum(1 for n in draw if 61 <= n <= 80),
        }

    def basic_statistics(self, top_n_triplets: int = 10) -> Dict[str, object]:
        total_draws = len(self.draw_numbers)
        counts = self.matrix.sum(axis=0)
        probs = counts / max(total_draws, 1)

        zone_per_draw = pd.DataFrame([self._zone_counts(d) for d in self.draw_numbers])
        zone_avg = zone_per_draw.mean().to_dict()
        zone_burst_ge7 = (zone_per_draw >= 7).sum().to_dict()
        zone_burst_ge8 = (zone_per_draw >= 8).sum().to_dict()

        big_small = []
        odd_even = []
        chain_counter = Counter({"2連": 0, "3連": 0, "4連以上": 0})
        triplets = Counter()

        for draw in self.draw_numbers:
            small = sum(1 for n in draw if n <= 40)
            big = 20 - small
            big_small.append((small, big))

            odd = sum(1 for n in draw if n % 2 == 1)
            even = 20 - odd
            odd_even.append((odd, even))

            runs = self._consecutive_runs(draw)
            for run in runs:
                if run == 2:
                    chain_counter["2連"] += 1
                elif run == 3:
                    chain_counter["3連"] += 1
                elif run >= 4:
                    chain_counter["4連以上"] += 1

            for combo in combinations(draw, 3):
                triplets[combo] += 1

        top_triplets = [
            {"numbers": list(nums), "count": c}
            for nums, c in triplets.most_common(top_n_triplets)
        ]

        return {
            "total_draws": total_draws,
            "number_total_counts": {str(i + 1): int(c) for i, c in enumerate(counts)},
            "number_probabilities": {str(i + 1): float(p) for i, p in enumerate(probs)},
            "zone_stats": {
                "average_per_draw": {k: float(v) for k, v in zone_avg.items()},
                "burst_ge_7": {k: int(v) for k, v in zone_burst_ge7.items()},
                "burst_ge_8": {k: int(v) for k, v in zone_burst_ge8.items()},
            },
            "big_small": {
                "per_draw": [{"small": s, "big": b} for s, b in big_small],
                "year_average": {
                    "small": float(np.mean([s for s, _ in big_small])),
                    "big": float(np.mean([b for _, b in big_small])),
                },
            },
            "odd_even": {
                "per_draw": [{"odd": o, "even": e} for o, e in odd_even],
                "year_average": {
                    "odd": float(np.mean([o for o, _ in odd_even])),
                    "even": float(np.mean([e for _, e in odd_even])),
                },
            },
            "consecutive_stats": dict(chain_counter),
            "top_triplets": top_triplets,
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

    def dynamic_analysis(
        self,
        recent_draws: Optional[Sequence[Sequence[int]]] = None,
        latest_issue: Optional[int] = None,
    ) -> Dict[str, object]:
        draws = list(recent_draws) if recent_draws is not None else self.draw_numbers
        latest_draw = draws[-1]
        latest_zones = self._zone_counts(latest_draw)
        window5 = self._momentum_scores(draws, 5)
        window10 = self._momentum_scores(draws, 10)
        trend = self._zone_trend(draws)
        board_type = self._classify_board(latest_zones, trend)

        resolved_latest_issue = (
            latest_issue if latest_issue is not None else int(self.df.iloc[-1]["期別"])
        )

        return {
            "latest_issue": resolved_latest_issue,
            "latest_draw": list(latest_draw),
            "momentum_last_5": window5,
            "momentum_last_10": window10,
            "zone_trend": trend,
            "board_type": board_type,
        }

    def _momentum_scores(
        self, draws: Sequence[Sequence[int]], window: int
    ) -> Dict[str, float]:
        sub_draws = list(draws)[-window:]
        sub_matrix = self._build_matrix(sub_draws)
        scores = sub_matrix.sum(axis=0) / max(len(sub_draws), 1)
        return {str(i + 1): float(v) for i, v in enumerate(scores)}

    def _zone_trend(self, draws: Sequence[Sequence[int]]) -> Dict[str, object]:
        recent = list(draws)[-10:]
        zones = [self._zone_counts(d) for d in recent]
        zone_df = pd.DataFrame(zones)
        recent_mean = zone_df.tail(5).mean()
        older_mean = zone_df.head(5).mean()

        compression_to_burst = {}
        burst_to_fall = {}
        for zone in ["A", "B", "C", "D"]:
            compression_to_burst[zone] = bool(
                older_mean[zone] <= 4 and recent_mean[zone] >= 6
            )
            burst_to_fall[zone] = bool(older_mean[zone] >= 7 and recent_mean[zone] <= 5)

        latest = zones[-1]
        balanced = all(v == 5 for v in latest.values())

        return {
            "compression_to_burst": compression_to_burst,
            "burst_to_fall": burst_to_fall,
            "is_balanced_5_5_5_5": balanced,
            "recent_zone_mean": {k: float(v) for k, v in recent_mean.items()},
        }

    def _classify_board(
        self, latest_zones: Dict[str, int], trend: Dict[str, object]
    ) -> str:
        counts = list(latest_zones.values())
        if any(c >= 8 for c in counts):
            return "爆發盤"
        if sum(1 for c in counts if c >= 6) >= 2:
            return "雙區震盪盤"
        if all(c == 5 for c in counts):
            return "均衡盤"
        if any(trend["burst_to_fall"].values()):
            return "修正盤"
        if latest_zones["B"] + latest_zones["C"] >= 12:
            return "中段主導盤"
        return "修正盤"

    def predict_next(
        self,
        recent_draws: Sequence[Sequence[int]],
        latest_issue: int,
        weights: ScoreWeights = ScoreWeights(),
        top_k: int = 20,
    ) -> Dict[str, object]:
        # 核心規則註解：predict_next 禁止直接使用全年資料當短期輸入，必須由外部傳入 recent 10–50 期。
        self._validate_recent_draws(recent_draws)
        short_window = len(recent_draws)
        target_issue = latest_issue + 1
        recent_matrix = self._build_matrix_from_recent(recent_draws)
        recent_freq = recent_matrix.mean(axis=0)

        dynamic = self.dynamic_analysis(
            recent_draws=recent_draws, latest_issue=latest_issue
        )
        board_type = dynamic["board_type"]
        zone_target = self._predict_zone_target(board_type, recent_draws)

        zone_weights = np.array(
            [self._zone_weight(i + 1, zone_target) for i in range(80)]
        )
        combo_scores, triplet_counter = self._combo_resonance_scores(recent_draws)

        score = (
            recent_freq * weights.alpha
            + zone_weights * weights.beta
            + self.long_term_probs * weights.gamma
            + combo_scores * weights.delta
        )

        ranking = np.argsort(score)[::-1] + 1
        selected = ranking[:top_k].tolist()

        # 中文註解：Top3 只代表下一期同一期內三號共現機率最高組合，避免誤解為跨多期分散預測。
        top_triplet = self._best_next_issue_triplet(
            selected, score, triplet_counter, recent_draws, weights
        )
        confidence = self._confidence_labels(score, selected, board_type)

        return {
            "latest_issue": latest_issue,
            "target_issue": target_issue,
            "board_type": board_type,
            "dynamic": {
                "zone_trend": dynamic["zone_trend"],
                "latest_draw": dynamic["latest_draw"],
            },
            "predicted_zone_counts": zone_target,
            "top3_triplet": top_triplet,
            "number_scores": {str(i + 1): float(v) for i, v in enumerate(score)},
            "predicted_numbers_top20": selected,
            "confidence": confidence,
            "weights": weights.__dict__,
            "short_window": short_window,
        }

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

    def _predict_zone_target(
        self, board_type: str, recent_draws: Sequence[Sequence[int]]
    ) -> Dict[str, int]:
        latest = self._zone_counts(list(recent_draws)[-1])
        target = latest.copy()

        if board_type == "爆發盤":
            burst_zone = max(latest, key=latest.get)
            if latest[burst_zone] >= 8:
                target[burst_zone] = max(4, latest[burst_zone] - 2)
        elif board_type == "均衡盤":
            burst_zone = max(
                ["A", "B", "C", "D"],
                key=lambda z: self._zone_recent_mean(z, recent_draws, 10),
            )
            target[burst_zone] = 7
            for z in target:
                if z != burst_zone:
                    target[z] = 13 // 3
        elif board_type == "雙區震盪盤":
            top_two = sorted(latest, key=latest.get, reverse=True)[:2]
            target[top_two[0]] = min(8, latest[top_two[0]] + 1)
            target[top_two[1]] = max(4, latest[top_two[1]] - 1)
        elif board_type == "中段主導盤":
            target["B"], target["C"] = 6, 6
            target["A"], target["D"] = 4, 4

        total = sum(target.values())
        if total != 20:
            target = self._normalize_zone_total(target, total)
        return target

    @staticmethod
    def _normalize_zone_total(target: Dict[str, int], total: int) -> Dict[str, int]:
        keys = ["A", "B", "C", "D"]
        while total > 20:
            k = max(keys, key=lambda x: target[x])
            if target[k] > 0:
                target[k] -= 1
                total -= 1
        while total < 20:
            k = min(keys, key=lambda x: target[x])
            target[k] += 1
            total += 1
        return target

    def _zone_recent_mean(
        self, zone: str, draws: Sequence[Sequence[int]], window: int
    ) -> float:
        selected = list(draws)[-window:]
        return float(np.mean([self._zone_counts(d)[zone] for d in selected]))

    @staticmethod
    def _zone_weight(number: int, target: Dict[str, int]) -> float:
        if 1 <= number <= 20:
            return target["A"] / 20
        if 21 <= number <= 40:
            return target["B"] / 20
        if 41 <= number <= 60:
            return target["C"] / 20
        return target["D"] / 20

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

    def _best_next_issue_triplet(
        self,
        numbers: Sequence[int],
        score: np.ndarray,
        triplet_counter: Counter,
        recent_draws: Sequence[Sequence[int]],
        weights: ScoreWeights,
    ) -> Dict[str, object]:
        pair_counter = Counter()
        for draw in recent_draws:
            for pair in combinations(draw, 2):
                pair_counter[tuple(sorted(pair))] += 1

        max_triplet = max(triplet_counter.values(), default=1)
        max_pair = max(pair_counter.values(), default=1)
        candidate_pool = list(numbers[: min(12, len(numbers))])

        best_combo: Tuple[int, int, int] = tuple(candidate_pool[:3])
        best_score = float("-inf")
        best_explain: Dict[str, object] = {}

        # 中文註解：以 short-term triplet、pair resonance、個別分數、區段適配混合評分。
        for combo in combinations(candidate_pool, 3):
            combo = tuple(sorted(combo))
            recent_triplet_count = int(triplet_counter.get(combo, 0))
            triplet_strength = recent_triplet_count / max_triplet
            pair_strength = np.mean(
                [
                    pair_counter.get(tuple(sorted((combo[0], combo[1]))), 0) / max_pair,
                    pair_counter.get(tuple(sorted((combo[0], combo[2]))), 0) / max_pair,
                    pair_counter.get(tuple(sorted((combo[1], combo[2]))), 0) / max_pair,
                ]
            )
            individual_strength = float(np.mean([score[n - 1] for n in combo]))
            zone_strength, zone_fit = self._triplet_zone_fit(combo)

            combined = (
                weights.alpha * triplet_strength
                + weights.beta * pair_strength
                + weights.gamma * individual_strength
                + weights.delta * zone_strength
            )
            if combined > best_score:
                best_score = combined
                best_combo = combo
                best_explain = {
                    "recent_triplet_count": recent_triplet_count,
                    "recent_pair_resonance": round(float(pair_strength), 4),
                    "zone_fit": zone_fit,
                    "blend_weights": weights.__dict__,
                }

        return {
            "numbers": list(best_combo),
            "score": round(float(best_score), 4),
            "explain": best_explain,
        }

    @staticmethod
    def _triplet_zone_fit(combo: Sequence[int]) -> Tuple[float, str]:
        zone_map = {"A": 0, "B": 0, "C": 0, "D": 0}
        for num in combo:
            if 1 <= num <= 20:
                zone_map["A"] += 1
            elif 21 <= num <= 40:
                zone_map["B"] += 1
            elif 41 <= num <= 60:
                zone_map["C"] += 1
            else:
                zone_map["D"] += 1

        dominant = sorted(zone_map.items(), key=lambda kv: kv[1], reverse=True)
        zone_fit = "/".join([zone for zone, cnt in dominant if cnt > 0]) + " 偏強"
        return dominant[0][1] / 3, zone_fit

    @staticmethod
    def _confidence_labels(
        score: np.ndarray, selected: Sequence[int], board_type: str
    ) -> Dict[str, str]:
        selected_scores = np.array([score[n - 1] for n in selected])
        high = float(np.percentile(selected_scores, 75))
        low = float(np.percentile(selected_scores, 25))
        return {
            "score_band": f"high>={high:.3f}, low<={low:.3f}",
            "momentum": "高動能" if high > 0.35 else "中性",
            "structure": f"{board_type} 結構支撐",
        }


@lru_cache(maxsize=1)
def get_analyzer() -> BingoAnalyzer:
    # 核心規則註解：改為 lazy load，只有 API 實際呼叫時才初始化年度資料。
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
    # 核心規則註解：未提供 recent 10–50 期資料時，直接拒絕預測。
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

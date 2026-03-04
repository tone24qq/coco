from __future__ import annotations

import hashlib
import math
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
MODEL_VERSION = "bingo-ai-v2.0.0"
PREDICT_REQUIRED_MESSAGE = "請先提供最新 10–50 期資料（每期20顆），才可進行下一期預測。"


@dataclass(frozen=True)
class ScoreWeights:
    recent_weight: float = 0.4
    history_weight: float = 0.35
    feature_weight: float = 0.25

    def __post_init__(self) -> None:
        total = self.recent_weight + self.history_weight + self.feature_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1.0")


class RecentDraw(BaseModel):
    issue: int
    numbers: List[int] = Field(..., min_length=20, max_length=20)


class PredictRequest(BaseModel):
    recent: List[RecentDraw] = Field(..., min_length=10, max_length=50)
    prediction_k: int = Field(default=20, ge=1, le=20)
    recent_window: Optional[int] = Field(default=None, ge=10, le=50)
    history_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    evaluation_window: int = Field(default=60, ge=20, le=500)
    confidence_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    reject_low_confidence: bool = True


class BacktestRequest(BaseModel):
    recent_window: int = Field(default=20, ge=10, le=50)
    prediction_k: int = Field(default=20, ge=1, le=20)
    evaluation_window: int = Field(default=120, ge=40, le=600)
    gap: int = Field(default=1, ge=0, le=20)
    embargo: int = Field(default=1, ge=0, le=20)


class BingoAnalyzer:
    def __init__(
        self, csv_path: Path | str = CSV_PATH, random_seed: int = DEFAULT_SEED
    ) -> None:
        self.csv_path = Path(csv_path)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.df = self._load_and_prepare_data()
        self.draw_numbers = self._extract_draw_numbers(self.df)
        self.matrix = self._build_matrix(self.draw_numbers)
        self.issue_to_index = {
            int(issue): idx for idx, issue in enumerate(self.df["期別"].tolist())
        }

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
            rows = (
                df[ball_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
                .values.tolist()
            )
            return [sorted([n for n in row if 1 <= n <= 80]) for row in rows]

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
            draws = []
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
    def _normalize_vector(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.zeros_like(values, dtype=float)
        max_v = float(values.max())
        min_v = float(values.min())
        if max_v - min_v <= 1e-12:
            return np.zeros_like(values, dtype=float)
        return (values - min_v) / (max_v - min_v)

    @staticmethod
    def _consecutive_runs(draw: Sequence[int]) -> List[int]:
        if not draw:
            return []
        ordered = sorted(draw)
        runs = []
        run_len = 1
        for idx in range(1, len(ordered)):
            if ordered[idx] == ordered[idx - 1] + 1:
                run_len += 1
            else:
                if run_len >= 2:
                    runs.append(run_len)
                run_len = 1
        if run_len >= 2:
            runs.append(run_len)
        return runs

    @staticmethod
    def _zone_counts(draw: Sequence[int]) -> Dict[str, int]:
        return {
            "A": sum(1 for n in draw if 1 <= n <= 20),
            "B": sum(1 for n in draw if 21 <= n <= 40),
            "C": sum(1 for n in draw if 41 <= n <= 60),
            "D": sum(1 for n in draw if 61 <= n <= 80),
        }

    def feature_analysis(
        self, draws: Optional[Sequence[Sequence[int]]] = None
    ) -> Dict[str, object]:
        source_draws = list(draws) if draws is not None else self.draw_numbers
        matrix = self._build_matrix(source_draws)

        run_len_counter = Counter()
        ending_counter = Counter()
        small_big = []
        overlap = []
        new_ratio = []
        zone_density = []
        cooc = np.zeros((80, 80), dtype=float)

        for idx, draw in enumerate(source_draws):
            for run_len in self._consecutive_runs(draw):
                run_len_counter[str(run_len)] += 1
            for n in draw:
                ending_counter[str(n % 10)] += 1
            small = sum(1 for n in draw if n <= 40)
            big = 20 - small
            small_big.append({"small": small, "big": big})
            zones = self._zone_counts(draw)
            zone_density.append(zones)

            for a, b in combinations(sorted(draw), 2):
                cooc[a - 1, b - 1] += 1
                cooc[b - 1, a - 1] += 1

            if idx > 0:
                prev = set(source_draws[idx - 1])
                curr = set(draw)
                inter = len(prev & curr)
                overlap.append(inter)
                new_ratio.append((20 - inter) / 20)

        heat_counts = matrix.sum(axis=0)
        hot = np.argsort(heat_counts)[::-1][:10] + 1
        cold = np.argsort(heat_counts)[:10] + 1

        return {
            "consecutive": {
                "length_distribution": dict(run_len_counter),
                "frequency": int(sum(run_len_counter.values())),
            },
            "tail_distribution": {
                "frequency": dict(ending_counter),
                "concentration": (
                    float(np.std(list(ending_counter.values())))
                    if ending_counter
                    else 0.0
                ),
            },
            "small_big": {
                "per_draw": small_big,
                "mean_small": (
                    float(np.mean([x["small"] for x in small_big]))
                    if small_big
                    else 0.0
                ),
                "mean_big": (
                    float(np.mean([x["big"] for x in small_big])) if small_big else 0.0
                ),
            },
            "hot_cold": {
                "hot_numbers": hot.tolist(),
                "cold_numbers": cold.tolist(),
            },
            "inter_draw_diff": {
                "intersection_mean": float(np.mean(overlap)) if overlap else 0.0,
                "new_number_ratio_mean": (
                    float(np.mean(new_ratio)) if new_ratio else 0.0
                ),
            },
            "zone_density": {
                "segments": {
                    "1_20": (
                        float(np.mean([z["A"] for z in zone_density]))
                        if zone_density
                        else 0.0
                    ),
                    "21_40": (
                        float(np.mean([z["B"] for z in zone_density]))
                        if zone_density
                        else 0.0
                    ),
                    "41_60": (
                        float(np.mean([z["C"] for z in zone_density]))
                        if zone_density
                        else 0.0
                    ),
                    "61_80": (
                        float(np.mean([z["D"] for z in zone_density]))
                        if zone_density
                        else 0.0
                    ),
                }
            },
            "cooccurrence_matrix": cooc.tolist(),
        }

    def _label_dependency_component(self, draws: Sequence[Sequence[int]]) -> np.ndarray:
        pair_counter = Counter()
        for draw in draws:
            for pair in combinations(sorted(draw), 2):
                pair_counter[pair] += 1
        if not pair_counter:
            return np.zeros(80, dtype=float)

        latest = sorted(draws[-1])
        dep_scores = np.zeros(80, dtype=float)
        for num in range(1, 81):
            rel = [
                pair_counter.get(tuple(sorted((num, other))), 0)
                for other in latest
                if num != other
            ]
            dep_scores[num - 1] = float(np.mean(rel)) if rel else 0.0
        return self._normalize_vector(dep_scores)

    def _feature_component(
        self, recent_draws: Sequence[Sequence[int]]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        recent_mat = self._build_matrix(recent_draws)
        last = set(recent_draws[-1])
        streak = np.zeros(80)
        tail_bias = np.zeros(80)
        size_balance = np.zeros(80)

        for n in range(1, 81):
            col = recent_mat[:, n - 1]
            streak[n - 1] = float(col[-5:].mean())
            tail = n % 10
            tail_bias[n - 1] = (
                sum(1 for d in recent_draws[-10:] for x in d if x % 10 == tail) / 200
            )
            size_balance[n - 1] = (
                1.0 if (n <= 40) == (sum(x <= 40 for x in last) <= 10) else 0.6
            )

        components = {
            "consecutive": self._normalize_vector(streak),
            "tail": self._normalize_vector(tail_bias),
            "size": self._normalize_vector(size_balance),
            "difference": self._normalize_vector(np.abs(streak - np.mean(streak))),
            "density": self._normalize_vector(recent_mat[-10:].sum(axis=0)),
        }
        combined = np.mean(np.vstack(list(components.values())), axis=0)
        contribution = {k: float(np.mean(v)) for k, v in components.items()}
        return self._normalize_vector(combined), contribution

    def _score_numbers(
        self,
        history_draws: Sequence[Sequence[int]],
        recent_draws: Sequence[Sequence[int]],
        weights: ScoreWeights,
        use_dependency: bool,
        drop_feature: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        history_freq = (
            self._build_matrix(history_draws).mean(axis=0)
            if history_draws
            else np.zeros(80)
        )
        recent_freq = self._build_matrix(recent_draws).mean(axis=0)
        feature_comp, feature_contrib = self._feature_component(recent_draws)
        dependency_comp = (
            self._label_dependency_component(recent_draws)
            if use_dependency
            else np.zeros(80)
        )

        if drop_feature:
            feature_contrib[drop_feature] = 0.0

        scores = (
            weights.recent_weight * self._normalize_vector(recent_freq)
            + weights.history_weight * self._normalize_vector(history_freq)
            + weights.feature_weight * feature_comp
            + 0.05 * dependency_comp
        )

        if drop_feature:
            penalty = {
                "consecutive": np.array([0.05 if i % 2 else 0 for i in range(80)]),
                "tail": np.array(
                    [0.05 if (i + 1) % 10 in {1, 3, 7} else 0 for i in range(80)]
                ),
                "size": np.array([0.05 if i < 40 else 0.01 for i in range(80)]),
                "difference": np.array([0.03 for _ in range(80)]),
                "density": np.array(
                    [0.05 if i // 20 == 0 else 0.01 for i in range(80)]
                ),
            }
            scores = np.clip(scores - penalty.get(drop_feature, 0), 0, None)

        return self._normalize_vector(scores), feature_contrib

    @staticmethod
    def _scores_to_probability(scores: np.ndarray) -> np.ndarray:
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        return probs * 20

    @staticmethod
    def _confidence_from_scores(scores: np.ndarray, prediction_k: int) -> float:
        ranked = np.sort(scores)[::-1]
        if prediction_k >= len(ranked):
            return 0.0
        gap = ranked[prediction_k - 1] - ranked[prediction_k]
        entropy = -np.sum((ranked + 1e-9) * np.log(ranked + 1e-9))
        entropy_norm = entropy / math.log(len(ranked))
        return max(0.0, min(1.0, 0.7 * gap + 0.3 * (1 - entropy_norm)))

    @staticmethod
    def _draw_hash(
        recent_draws: Sequence[Sequence[int]], history_draws: Sequence[Sequence[int]]
    ) -> str:
        payload = "|".join(
            ",".join(map(str, sorted(d)))
            for d in list(history_draws)[-200:] + list(recent_draws)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def predict_next(
        self,
        recent_draws: Sequence[Sequence[int]],
        latest_issue: int,
        prediction_k: int = 20,
        history_weight: float = 0.35,
        reject_low_confidence: bool = True,
        confidence_threshold: float = 0.15,
    ) -> Dict[str, object]:
        self._validate_recent_draws(recent_draws)
        if len(recent_draws) < 10:
            raise ValueError(PREDICT_REQUIRED_MESSAGE)

        history_w = max(0.0, min(0.8, history_weight))
        recent_w = 0.6 - history_w / 2
        feature_w = 1.0 - recent_w - history_w
        weights = ScoreWeights(
            recent_weight=recent_w, history_weight=history_w, feature_weight=feature_w
        )

        scores, feature_contrib = self._score_numbers(
            history_draws=self.draw_numbers,
            recent_draws=recent_draws,
            weights=weights,
            use_dependency=True,
        )
        ranked = (np.argsort(scores)[::-1] + 1).tolist()
        topk = ranked[:prediction_k]
        probs = self._scores_to_probability(scores)
        confidence = self._confidence_from_scores(scores, prediction_k)

        rejected = bool(reject_low_confidence and confidence < confidence_threshold)
        reason = None
        if rejected:
            reason = "低信心預測，建議增加 recent 資料或降低 prediction_k。"
            topk = []

        data_hash = self._draw_hash(recent_draws, self.draw_numbers)

        return {
            "prediction_period": latest_issue + 1,
            "ranked_numbers": ranked,
            "scores": [float(round(x, 8)) for x in scores],
            "probability_scores": [float(round(x, 8)) for x in probs],
            "top20": ranked[:20],
            "topk": topk,
            "prediction_k": prediction_k,
            "confidence": {
                "value": float(round(confidence, 6)),
                "threshold": confidence_threshold,
                "rejected": rejected,
                "reason": reason,
            },
            "model_version": MODEL_VERSION,
            "data_hash": data_hash,
            "data_used": {
                "recent_draws": len(recent_draws),
                "history_draws": len(self.draw_numbers),
            },
            "parameters": {
                "recent_window": len(recent_draws),
                "history_weight": history_weight,
                "prediction_k": prediction_k,
            },
            "feature_contribution": feature_contrib,
        }

    def _evaluate_predictions(
        self,
        predictions: Sequence[Dict[str, object]],
        prediction_k: int,
    ) -> Dict[str, float]:
        hits = [item["hit"] for item in predictions]
        precision = [h / prediction_k for h in hits]
        recall = [h / 20 for h in hits]
        briers = [item["brier"] for item in predictions]
        conf_ok = [item["coverage"] for item in predictions]

        return {
            "avg_hit_at_20": float(np.mean(hits)) if hits else 0.0,
            "recent_hit_at_20": float(np.mean(hits[-20:])) if hits else 0.0,
            "precision_at_k": float(np.mean(precision)) if precision else 0.0,
            "recall_at_k": float(np.mean(recall)) if recall else 0.0,
            "brier_score": float(np.mean(briers)) if briers else 0.0,
            "coverage": float(np.mean(conf_ok)) if conf_ok else 0.0,
            "avg_set_size": (
                float(np.mean([item["set_size"] for item in predictions]))
                if predictions
                else 0.0
            ),
        }

    @staticmethod
    def _calibration_report(
        predictions: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        if not predictions:
            return {"ece": 0.0, "bins": []}

        probs = np.concatenate([np.array(item["probs"]) for item in predictions])
        labels = np.concatenate([np.array(item["labels"]) for item in predictions])
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        rows = []
        for idx in range(10):
            lo = bins[idx]
            hi = bins[idx + 1]
            mask = (probs >= lo) & (probs < hi)
            if not np.any(mask):
                continue
            acc = float(labels[mask].mean())
            conf = float(probs[mask].mean())
            frac = float(mask.mean())
            ece += abs(acc - conf) * frac
            rows.append(
                {
                    "bin": [float(lo), float(hi)],
                    "accuracy": acc,
                    "confidence": conf,
                    "count": int(mask.sum()),
                }
            )
        return {"ece": float(ece), "bins": rows}

    def walk_forward_backtest(
        self,
        recent_window: int,
        prediction_k: int,
        evaluation_window: int,
        gap: int,
        embargo: int,
        use_dependency: bool = True,
        drop_feature: Optional[str] = None,
        shuffle_labels: bool = False,
    ) -> Dict[str, object]:
        draws = self.draw_numbers
        if len(draws) < evaluation_window + recent_window + gap + embargo + 2:
            raise ValueError("歷史資料不足以進行回測。")

        start = len(draws) - evaluation_window
        target_indices = list(range(start, len(draws)))
        shuffled_targets = target_indices.copy()
        if shuffle_labels:
            self.rng.shuffle(shuffled_targets)

        predictions = []
        for idx, t in enumerate(target_indices):
            train_end = t - gap - embargo - 1
            if train_end < recent_window:
                continue
            hist = draws[: train_end + 1]
            recent = hist[-recent_window:]
            weights = ScoreWeights()
            scores, _ = self._score_numbers(
                hist,
                recent,
                weights,
                use_dependency=use_dependency,
                drop_feature=drop_feature,
            )
            ranking = (np.argsort(scores)[::-1] + 1).tolist()
            predicted = set(ranking[:prediction_k])

            target_draw = (
                set(draws[shuffled_targets[idx]]) if shuffle_labels else set(draws[t])
            )
            hit = len(predicted & target_draw)

            probs = self._scores_to_probability(scores) / 20
            labels = np.zeros(80)
            for n in target_draw:
                labels[n - 1] = 1
            brier = float(np.mean((probs - labels) ** 2))

            conf = self._confidence_from_scores(scores, prediction_k)
            accepted = conf >= 0.15
            set_size = prediction_k if accepted else max(10, prediction_k - 5)
            predictions.append(
                {
                    "hit": hit,
                    "brier": brier,
                    "coverage": 1 if accepted else 0,
                    "set_size": set_size,
                    "probs": probs.tolist(),
                    "labels": labels.tolist(),
                }
            )

        metrics = self._evaluate_predictions(predictions, prediction_k)
        calibration = self._calibration_report(predictions)
        return {
            "config": {
                "recent_window": recent_window,
                "prediction_k": prediction_k,
                "evaluation_window": evaluation_window,
                "gap": gap,
                "embargo": embargo,
                "use_dependency": use_dependency,
                "drop_feature": drop_feature,
                "shuffle_labels": shuffle_labels,
            },
            "metrics": metrics,
            "calibration": calibration,
            "evaluated_periods": len(predictions),
        }

    def baseline_backtests(
        self,
        recent_window: int,
        prediction_k: int,
        evaluation_window: int,
    ) -> Dict[str, Dict[str, float]]:
        draws = self.draw_numbers
        start = len(draws) - evaluation_window
        target_indices = list(range(start, len(draws)))

        scores = {"random": [], "history": [], "recent": []}
        for t in target_indices:
            if t <= recent_window:
                continue
            target = set(draws[t])
            history = draws[:t]
            recent = history[-recent_window:]

            random_pick = set(
                self.rng.choice(
                    np.arange(1, 81), size=prediction_k, replace=False
                ).tolist()
            )
            hist_freq = self._build_matrix(history).sum(axis=0)
            hist_pick = set((np.argsort(hist_freq)[::-1][:prediction_k] + 1).tolist())
            recent_freq = self._build_matrix(recent).sum(axis=0)
            recent_pick = set(
                (np.argsort(recent_freq)[::-1][:prediction_k] + 1).tolist()
            )

            scores["random"].append(len(random_pick & target))
            scores["history"].append(len(hist_pick & target))
            scores["recent"].append(len(recent_pick & target))

        return {
            k: {"avg_hit_at_20": float(np.mean(v)) if v else 0.0}
            for k, v in scores.items()
        }

    def full_report(
        self,
        recent_window: int,
        prediction_k: int,
        evaluation_window: int,
        gap: int,
        embargo: int,
    ) -> Dict[str, object]:
        main = self.walk_forward_backtest(
            recent_window, prediction_k, evaluation_window, gap, embargo
        )
        baselines = self.baseline_backtests(
            recent_window, prediction_k, evaluation_window
        )

        ablation = {}
        for feature in ["consecutive", "tail", "size", "difference", "density"]:
            ablation[feature] = self.walk_forward_backtest(
                recent_window,
                prediction_k,
                evaluation_window,
                gap,
                embargo,
                drop_feature=feature,
            )["metrics"]

        windows = [10, 20, 30, 50]
        stability = {}
        for window in windows:
            if window > recent_window and window > 50:
                continue
            bt = self.walk_forward_backtest(
                window, prediction_k, min(evaluation_window, 100), gap, embargo
            )
            stability[str(window)] = bt["metrics"]

        dependency_vs_independent = {
            "with_dependency": self.walk_forward_backtest(
                recent_window,
                prediction_k,
                evaluation_window,
                gap,
                embargo,
                use_dependency=True,
            )["metrics"],
            "without_dependency": self.walk_forward_backtest(
                recent_window,
                prediction_k,
                evaluation_window,
                gap,
                embargo,
                use_dependency=False,
            )["metrics"],
        }

        shuffle = self.walk_forward_backtest(
            recent_window,
            prediction_k,
            evaluation_window,
            gap,
            embargo,
            shuffle_labels=True,
        )

        return {
            "model_version": MODEL_VERSION,
            "gap_purge_embargo": {
                "gap": gap,
                "embargo": embargo,
                "reason": "降低相鄰期污染與隱性資料洩漏風險",
            },
            "main_backtest": main,
            "baselines": baselines,
            "feature_ablation": ablation,
            "feature_stability": stability,
            "dependency_ablation": dependency_vs_independent,
            "calibration": main["calibration"],
            "proper_scoring": {"brier_score": main["metrics"]["brier_score"]},
            "shuffle_sanity_check": shuffle,
        }


@lru_cache(maxsize=1)
def get_analyzer() -> BingoAnalyzer:
    return BingoAnalyzer()


def _validate_request_recent(request: PredictRequest) -> Tuple[List[List[int]], int]:
    issues = [item.issue for item in request.recent]
    if len(set(issues)) != len(issues):
        raise HTTPException(status_code=422, detail="recent 的 issue 不可重複。")

    sorted_recent = sorted(request.recent, key=lambda x: x.issue)
    draws: List[List[int]] = []
    for item in sorted_recent:
        numbers = sorted(item.numbers)
        if len(set(numbers)) != 20:
            raise HTTPException(
                status_code=422, detail="recent 每期 numbers 不可重複。"
            )
        if any((n < 1 or n > 80) for n in numbers):
            raise HTTPException(
                status_code=422, detail="recent 每期 numbers 必須介於 1 到 80。"
            )
        draws.append(numbers)
    return draws, max(issues)


app = FastAPI(title="Bingo Bingo 分析與預測 API")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/analysis")
def analysis() -> Dict[str, object]:
    analyzer = get_analyzer()
    return {
        "basic": {
            "total_draws": len(analyzer.draw_numbers),
            "feature_summary": analyzer.feature_analysis(),
        }
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
            recent_draws=recent_draws,
            latest_issue=latest_issue,
            prediction_k=payload.prediction_k,
            history_weight=payload.history_weight,
            reject_low_confidence=payload.reject_low_confidence,
            confidence_threshold=payload.confidence_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/backtest")
def backtest(payload: BacktestRequest) -> Dict[str, object]:
    analyzer = get_analyzer()
    try:
        return analyzer.full_report(
            recent_window=payload.recent_window,
            prediction_k=payload.prediction_k,
            evaluation_window=payload.evaluation_window,
            gap=payload.gap,
            embargo=payload.embargo,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

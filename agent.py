from __future__ import annotations

import json
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

CSV_PATH = Path(__file__).resolve().parent / "賓果賓果_2026.csv"
DEFAULT_SEED = 42
PREDICT_REQUIRED_MESSAGE = "請先提供最新 10–50 期資料（每期20顆），才可進行下一期預測。"


@dataclass(frozen=True)
class ScoreWeights:
    recent_weight: float = 0.2
    history_similar_weight: float = 0.5
    other_weight: float = 0.3

    def __post_init__(self) -> None:
        total = self.recent_weight + self.history_similar_weight + self.other_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1.0")


class RecentDraw(BaseModel):
    issue: int
    numbers: List[int] = Field(..., min_length=20, max_length=20)


class PredictRequest(BaseModel):
    recent: List[RecentDraw] = Field(..., min_length=10, max_length=50)
    top_k: int = Field(default=20, ge=1, le=20)


class BacktestRequest(BaseModel):
    windows: List[int] = Field(default_factory=lambda: [50, 100, 200])
    alphas: List[float] = Field(default_factory=lambda: [0.9, 0.95, 0.98])
    lambdas: List[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0])
    recent_n: int = Field(default=200, ge=20)
    candidate_pool_size: int = Field(default=18, ge=8, le=30)
    random_runs: int = Field(default=500, ge=100, le=5000)
    max_steps: Optional[int] = Field(default=None, ge=1)
    output_dir: str = Field(default="artifacts")


class HitCountBacktestRequest(BaseModel):
    min_train_size: int = Field(default=50, ge=20)
    knn_k: int = Field(default=15, ge=3, le=100)
    momentum_short: int = Field(default=5, ge=2, le=30)
    momentum_long: int = Field(default=20, ge=5, le=100)
    output_dir: str = Field(default="artifacts")


class BingoAnalyzer:
    def __init__(
        self, csv_path: Path | str = CSV_PATH, random_seed: int = DEFAULT_SEED
    ) -> None:
        self.csv_path = Path(csv_path)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.df = self._load_and_prepare_data()
        self.draw_numbers: List[List[int]] = self._extract_draw_numbers(self.df)
        self.matrix = self._build_matrix(self.draw_numbers)
        self.issue_to_index = {
            int(issue): idx for idx, issue in enumerate(self.df["期別"].tolist())
        }

    def run_hitcount_backtest(
        self, request: Optional[HitCountBacktestRequest] = None
    ) -> Dict[str, object]:
        cfg = request or HitCountBacktestRequest()
        draws = self.draw_numbers
        if len(draws) <= cfg.min_train_size + 1:
            raise ValueError("資料量不足，無法進行 walk-forward 回測。")

        rows: List[Dict[str, object]] = []
        for t in range(cfg.min_train_size - 1, len(draws) - 1):
            train_draws = draws[: t + 1]
            latest_draw = train_draws[-1]
            target_draw = draws[t + 1]
            issue_t = int(self.df.iloc[t]["期別"])
            issue_t1 = int(self.df.iloc[t + 1]["期別"])

            model_preds = {
                "markov_transition": self._predict_top20_from_prob(
                    self._markov_transition_probability(train_draws, latest_draw)
                ),
                "similar_knn": self._predict_top20_from_prob(
                    self._knn_next_probability(train_draws, k=cfg.knn_k)
                ),
                "short_momentum": self._predict_top20_from_prob(
                    self._short_momentum_probability(
                        train_draws,
                        short_window=cfg.momentum_short,
                        long_window=cfg.momentum_long,
                    )
                ),
            }

            for method, pred in model_preds.items():
                hit = len(set(pred) & set(target_draw))
                rows.append(
                    {
                        "issue_t": issue_t,
                        "issue_t1": issue_t1,
                        "method": method,
                        "predicted_top20": "-".join(map(str, pred)),
                        "target_top20": "-".join(map(str, sorted(target_draw))),
                        "hit_count": hit,
                    }
                )

        detail_df = pd.DataFrame(rows)
        summary_df = (
            detail_df.groupby("method", as_index=False)["hit_count"]
            .agg(["mean", "std", "max"])  # type: ignore[arg-type]
            .reset_index()
            .rename(
                columns={
                    "mean": "avg_hit_count",
                    "std": "hit_count_std",
                    "max": "best_hit_count",
                }
            )
            .sort_values(["avg_hit_count", "best_hit_count"], ascending=[False, False])
        )
        best_method = str(summary_df.iloc[0]["method"])

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        detail_path = output_dir / "hitcount_walkforward_detail.csv"
        summary_path = output_dir / "hitcount_walkforward_summary.csv"
        config_path = output_dir / "hitcount_best.json"
        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        best_config = {
            "best_method": best_method,
            "min_train_size": cfg.min_train_size,
            "knn_k": cfg.knn_k,
            "momentum_short": cfg.momentum_short,
            "momentum_long": cfg.momentum_long,
            "summary": summary_df.to_dict(orient="records"),
            "guardrail": "walk-forward only, strictly no look-ahead",
        }
        config_path.write_text(
            json.dumps(best_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "best_method": best_method,
            "summary": best_config["summary"],
            "output_files": {
                "detail": str(detail_path),
                "summary": str(summary_path),
                "best_config": str(config_path),
            },
        }

    @staticmethod
    def _predict_top20_from_prob(prob: np.ndarray) -> List[int]:
        ranking = np.argsort(prob)[::-1] + 1
        return ranking[:20].tolist()

    def _markov_transition_probability(
        self,
        train_draws: Sequence[Sequence[int]],
        latest_draw: Sequence[int],
    ) -> np.ndarray:
        trans = np.zeros((80, 80), dtype=float)
        for i in range(len(train_draws) - 1):
            curr = train_draws[i]
            nxt = train_draws[i + 1]
            for a in curr:
                for b in nxt:
                    trans[a - 1, b - 1] += 1.0

        row_sums = trans.sum(axis=1, keepdims=True)
        trans = np.divide(trans, np.maximum(row_sums, 1.0), where=row_sums > 0)
        if not latest_draw:
            return np.ones(80, dtype=float) / 80
        pred = trans[np.array(latest_draw) - 1].mean(axis=0)
        return pred

    def _knn_next_probability(
        self, train_draws: Sequence[Sequence[int]], k: int
    ) -> np.ndarray:
        if len(train_draws) < 2:
            return np.ones(80, dtype=float) / 80

        latest = set(train_draws[-1])
        candidates: List[Tuple[float, int]] = []
        for i in range(len(train_draws) - 1):
            hist = set(train_draws[i])
            inter = len(latest & hist)
            union = len(latest | hist)
            sim = inter / union if union else 0.0
            candidates.append((sim, i))
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        probs = np.zeros(80, dtype=float)
        chosen = candidates[: min(k, len(candidates))]
        total_w = sum(sim for sim, _ in chosen)
        if total_w <= 0:
            return np.ones(80, dtype=float) / 80
        for sim, idx in chosen:
            for n in train_draws[idx + 1]:
                probs[n - 1] += sim
        return probs / total_w

    def _short_momentum_probability(
        self,
        train_draws: Sequence[Sequence[int]],
        short_window: int,
        long_window: int,
    ) -> np.ndarray:
        matrix = self._build_matrix(train_draws)
        short_freq = matrix[-short_window:].mean(axis=0)
        long_freq = matrix[-long_window:].mean(axis=0)
        momentum = np.maximum(short_freq - long_freq, 0)
        score = 0.7 * short_freq + 0.3 * momentum
        total = score.sum()
        if total <= 1e-12:
            return np.ones(80, dtype=float) / 80
        return score / total

    def predict_online_top20(
        self, recent_draws: Sequence[Sequence[int]]
    ) -> Dict[str, object]:
        self._validate_recent_draws(recent_draws)
        cfg_path = Path("artifacts") / "hitcount_best.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            method = str(cfg["best_method"])
        else:
            method = "short_momentum"
            cfg = {
                "knn_k": 10,
                "momentum_short": min(5, len(recent_draws)),
                "momentum_long": min(20, len(recent_draws)),
                "summary": [],
            }

        if method == "markov_transition":
            prob = self._markov_transition_probability(recent_draws, recent_draws[-1])
        elif method == "similar_knn":
            prob = self._knn_next_probability(recent_draws, k=int(cfg["knn_k"]))
        else:
            prob = self._short_momentum_probability(
                recent_draws,
                short_window=int(cfg["momentum_short"]),
                long_window=int(cfg["momentum_long"]),
            )
        top20 = self._predict_top20_from_prob(prob)
        top3 = top20[:3]
        return {
            "online_method": method,
            "predicted_numbers_top20": top20,
            "top3": top3,
            "backtest_summary": cfg.get("summary", []),
        }

    def run_top3_backtest(
        self, request: Optional[BacktestRequest] = None
    ) -> Dict[str, object]:
        cfg = request or BacktestRequest()
        if not self.draw_numbers:
            raise ValueError("無可用開獎資料。")

        methods = ["random", "freq_only", "pair_only", "hybrid"]
        experiment_rows: List[Dict[str, object]] = []

        for window in sorted(set(cfg.windows)):
            for alpha in sorted(set(cfg.alphas)):
                for lam in sorted(set(cfg.lambdas)):
                    detail = self._walk_forward_backtest(
                        window=window,
                        alpha=alpha,
                        lam=lam,
                        candidate_pool_size=cfg.candidate_pool_size,
                        max_steps=cfg.max_steps,
                        random_runs=cfg.random_runs,
                    )
                    for method in methods:
                        stats = self._summarize_detail(
                            detail, method=method, recent_n=cfg.recent_n
                        )
                        experiment_rows.append(
                            {
                                "method": method,
                                "window": window,
                                "alpha": alpha,
                                "lambda": lam,
                                "random_runs": cfg.random_runs,
                                **stats,
                            }
                        )
        experiments_df = pd.DataFrame(experiment_rows)
        model_df = experiments_df[
            experiments_df["method"].isin(["freq_only", "pair_only", "hybrid"])
        ]
        if model_df.empty:
            raise ValueError("無法找到有效參數組合。")

        best_overall_row = model_df.sort_values(
            [
                "overall_triple_hit_rate",
                "overall_precision_at_3",
                "recent_triple_hit_rate",
            ],
            ascending=[False, False, False],
        ).iloc[0]
        best_recent_row = model_df.sort_values(
            [
                "recent_triple_hit_rate",
                "recent_precision_at_3",
                "overall_triple_hit_rate",
            ],
            ascending=[False, False, False],
        ).iloc[0]

        best_detail = self._walk_forward_backtest(
            window=int(best_recent_row["window"]),
            alpha=float(best_recent_row["alpha"]),
            lam=float(best_recent_row["lambda"]),
            candidate_pool_size=cfg.candidate_pool_size,
            max_steps=cfg.max_steps,
            random_runs=cfg.random_runs,
        )

        output_root = Path(cfg.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        detail_path = output_root / "backtest_detail.csv"
        experiments_path = output_root / "experiments.csv"
        best_config_path = output_root / "best_config.json"
        report_path = output_root / "report.md"

        pd.DataFrame(best_detail).to_csv(detail_path, index=False)
        experiments_df.sort_values(
            ["method", "overall_triple_hit_rate", "overall_precision_at_3"],
            ascending=[True, False, False],
        ).to_csv(experiments_path, index=False)

        best_config = {
            "best_overall": {
                "method": str(best_overall_row["method"]),
                "window": int(best_overall_row["window"]),
                "alpha": float(best_overall_row["alpha"]),
                "lambda": float(best_overall_row["lambda"]),
                "overall_triple_hit_rate": float(
                    best_overall_row["overall_triple_hit_rate"]
                ),
                "recent_triple_hit_rate": float(
                    best_overall_row["recent_triple_hit_rate"]
                ),
            },
            "best_recent": {
                "method": str(best_recent_row["method"]),
                "window": int(best_recent_row["window"]),
                "alpha": float(best_recent_row["alpha"]),
                "lambda": float(best_recent_row["lambda"]),
                "overall_triple_hit_rate": float(
                    best_recent_row["overall_triple_hit_rate"]
                ),
                "recent_triple_hit_rate": float(
                    best_recent_row["recent_triple_hit_rate"]
                ),
            },
            "candidate_pool_size": cfg.candidate_pool_size,
            "recent_n": cfg.recent_n,
            "random_runs": cfg.random_runs,
            "max_steps": cfg.max_steps,
        }
        best_config_path.write_text(
            json.dumps(best_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._write_report(
            report_path=report_path,
            experiments=experiments_df,
            best_detail_df=pd.DataFrame(best_detail),
            best_config=best_config,
            recent_n=cfg.recent_n,
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

    def _walk_forward_backtest(
        self,
        window: int,
        alpha: float,
        lam: float,
        candidate_pool_size: int,
        max_steps: Optional[int],
        random_runs: int,
    ) -> List[Dict[str, object]]:
        details: List[Dict[str, object]] = []
        if window <= 0 or window >= len(self.draw_numbers):
            return details

        all_indices = list(range(window, len(self.draw_numbers)))
        if max_steps is not None:
            all_indices = all_indices[-max_steps:]

        for idx in all_indices:
            train = self.draw_numbers[idx - window : idx]
            y_t = sorted(self.draw_numbers[idx])
            issue = int(self.df.iloc[idx]["期別"])

            p_scores, pair_scores = self._build_scores(train, alpha=alpha)
            random_preds = [
                self._predict_random(train, issue, run_id=i) for i in range(random_runs)
            ]
            random_hits = np.array(
                [len(set(pred) & set(y_t)) for pred in random_preds], dtype=float
            )
            pred_hot = self._predict_hot_only(p_scores)
            pred_pair = self._predict_pair_only(pair_scores, p_scores)
            pred_pair_aware = self._predict_pair_aware(
                p_scores, pair_scores, lam, candidate_pool_size
            )

            random_pred_mean = tuple(
                sorted(np.mean(np.asarray(random_preds), axis=0).astype(int).tolist())
            )
            random_hit_mean = float(np.mean(random_hits))
            random_hit_std = float(np.std((random_hits == 3).astype(float)))
            random_precision = random_hits / 3
            random_precision_std = float(np.std(random_precision))

            for method, pred in [
                ("random", random_pred_mean),
                ("freq_only", pred_hot),
                ("pair_only", pred_pair),
                ("hybrid", pred_pair_aware),
            ]:
                hit_count = len(set(pred) & set(y_t))
                if method == "random":
                    hit_count = random_hit_mean
                    triple_hit_t = float(np.mean((random_hits == 3).astype(float)))
                    precision_at_3 = float(np.mean(random_precision))
                    triple_hit_t_std = random_hit_std
                    precision_at_3_std = random_precision_std
                else:
                    triple_hit_t = float(hit_count == 3)
                    precision_at_3 = float(hit_count / 3)
                    triple_hit_t_std = 0.0
                    precision_at_3_std = 0.0
                details.append(
                    {
                        "issue": issue,
                        "method": method,
                        "window": window,
                        "alpha": alpha,
                        "lambda": lam,
                        "P_t": "-".join(map(str, sorted(pred))),
                        "Y_t": "-".join(map(str, y_t)),
                        "hit_count_t": float(hit_count),
                        "triple_hit_t": triple_hit_t,
                        "precision_at_3": precision_at_3,
                        "precision_at_3_std": precision_at_3_std,
                        "triple_hit_t_std": triple_hit_t_std,
                        "recall_at_3": float(hit_count / 20),
                    }
                )
        return details

    def _build_scores(
        self, train_draws: Sequence[Sequence[int]], alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        p_scores = np.zeros(80, dtype=float)
        pair_scores = np.zeros((80, 80), dtype=float)
        total_weight = 0.0

        for recency, draw in enumerate(reversed(train_draws)):
            w = alpha**recency
            total_weight += w
            for n in draw:
                p_scores[n - 1] += w
            for i, j in combinations(sorted(draw), 2):
                pair_scores[i - 1, j - 1] += w
                pair_scores[j - 1, i - 1] += w

        if total_weight > 0:
            p_scores /= total_weight
            pair_scores /= total_weight
        return p_scores, pair_scores

    def _predict_random(
        self, train_draws: Sequence[Sequence[int]], issue: int, run_id: int = 0
    ) -> Tuple[int, int, int]:
        seed = self.random_seed + issue + len(train_draws) + run_id
        rng = np.random.default_rng(seed)
        numbers = sorted(rng.choice(np.arange(1, 81), size=3, replace=False).tolist())
        return tuple(numbers)

    @staticmethod
    def _predict_hot_only(p_scores: np.ndarray) -> Tuple[int, int, int]:
        top = np.argsort(p_scores)[::-1][:3] + 1
        return tuple(sorted(top.tolist()))

    @staticmethod
    def _predict_pair_only(
        pair_scores: np.ndarray, p_scores: np.ndarray
    ) -> Tuple[int, int, int]:
        upper = np.triu(pair_scores, k=1)
        best_idx = np.argmax(upper)
        i, j = np.unravel_index(best_idx, upper.shape)
        pair = {i + 1, j + 1}
        candidates = np.argsort(p_scores)[::-1] + 1
        third = next(
            (int(n) for n in candidates if int(n) not in pair), int(candidates[0])
        )
        return tuple(sorted([i + 1, j + 1, third]))

    @staticmethod
    def _predict_pair_aware(
        p_scores: np.ndarray,
        pair_scores: np.ndarray,
        lam: float,
        candidate_pool_size: int,
    ) -> Tuple[int, int, int]:
        eps = 1e-9
        candidates = (np.argsort(p_scores)[::-1][:candidate_pool_size] + 1).tolist()
        best_combo = tuple(sorted(candidates[:3]))
        best_score = float("-inf")

        for combo in combinations(candidates, 3):
            a, b, c = combo
            score = (
                np.log(p_scores[a - 1] + eps)
                + np.log(p_scores[b - 1] + eps)
                + np.log(p_scores[c - 1] + eps)
                + lam
                * (
                    pair_scores[a - 1, b - 1]
                    + pair_scores[a - 1, c - 1]
                    + pair_scores[b - 1, c - 1]
                )
            )
            if score > best_score:
                best_score = float(score)
                best_combo = tuple(sorted((a, b, c)))
        return best_combo

    @staticmethod
    def _summarize_detail(
        rows: Sequence[Dict[str, object]], method: str, recent_n: int
    ) -> Dict[str, float]:
        df = pd.DataFrame([r for r in rows if r["method"] == method])
        if df.empty:
            return {
                "periods": 0,
                "overall_triple_hit_rate": 0.0,
                "overall_precision_at_3": 0.0,
                "overall_recall_at_3": 0.0,
                "recent_periods": 0,
                "recent_triple_hit_rate": 0.0,
                "recent_precision_at_3": 0.0,
                "recent_recall_at_3": 0.0,
                "triple_hit_rate_std": 0.0,
                "random_triple_hit_rate_mean": 0.0,
                "random_triple_hit_rate_std": 0.0,
                "random_precision_at_3_mean": 0.0,
                "random_precision_at_3_std": 0.0,
            }
        recent_df = df.tail(recent_n)
        triple_hit_rate = float(df["triple_hit_t"].mean())
        precision_at_3 = float(df["precision_at_3"].mean())
        triple_std = float(df["triple_hit_t_std"].mean()) if method == "random" else 0.0
        precision_std = (
            float(df["precision_at_3_std"].mean()) if method == "random" else 0.0
        )
        return {
            "periods": int(len(df)),
            "overall_triple_hit_rate": triple_hit_rate,
            "overall_precision_at_3": precision_at_3,
            "overall_recall_at_3": float(df["recall_at_3"].mean()),
            "recent_periods": int(len(recent_df)),
            "recent_triple_hit_rate": float(recent_df["triple_hit_t"].mean()),
            "recent_precision_at_3": float(recent_df["precision_at_3"].mean()),
            "recent_recall_at_3": float(recent_df["recall_at_3"].mean()),
            "triple_hit_rate_std": triple_std,
            "random_triple_hit_rate_mean": (
                triple_hit_rate if method == "random" else 0.0
            ),
            "random_triple_hit_rate_std": triple_std,
            "random_precision_at_3_mean": precision_at_3 if method == "random" else 0.0,
            "random_precision_at_3_std": precision_std,
        }

    def _write_report(
        self,
        report_path: Path,
        experiments: pd.DataFrame,
        best_detail_df: pd.DataFrame,
        best_config: Dict[str, object],
        recent_n: int,
    ) -> None:
        random_rows = experiments[experiments["method"] == "random"]
        random_summary = random_rows.sort_values(
            "overall_triple_hit_rate", ascending=False
        ).head(1)
        random_top = (
            random_summary.iloc[0].to_dict() if not random_summary.empty else {}
        )
        recent_hybrid = best_detail_df[best_detail_df["method"] == "hybrid"].tail(
            recent_n
        )
        recent_rate = (
            float(recent_hybrid["triple_hit_t"].mean())
            if not recent_hybrid.empty
            else 0.0
        )

        best_overall = best_config["best_overall"]
        best_recent = best_config["best_recent"]

        content = (
            "# Top-3 同期三顆同出回測報告\n\n"
            f"- best_overall：{best_overall['method']} (W={best_overall['window']}, alpha={best_overall['alpha']}, lambda={best_overall['lambda']})\n"
            f"- best_overall triple_hit_rate：{best_overall['overall_triple_hit_rate']:.6f}\n"
            f"- best_recent：{best_recent['method']} (W={best_recent['window']}, alpha={best_recent['alpha']}, lambda={best_recent['lambda']})\n"
            f"- best_recent triple_hit_rate：{best_recent['recent_triple_hit_rate']:.6f}\n"
            f"- 最近 {recent_n} 期 triple_hit_rate：{recent_rate:.6f}\n"
            f"- random baseline triple_hit_rate：{random_top.get('random_triple_hit_rate_mean', 0):.6f}±{random_top.get('random_triple_hit_rate_std', 0):.6f}\n"
            f"- random baseline precision@3：{random_top.get('random_precision_at_3_mean', 0):.6f}±{random_top.get('random_precision_at_3_std', 0):.6f}\n"
        )
        report_path.write_text(content, encoding="utf-8")

    def predict_top3_with_best(
        self, recent_draws: Sequence[Sequence[int]], use: str = "recent"
    ) -> Dict[str, object]:
        config_path = Path("artifacts") / "best_config.json"
        if not config_path.exists():
            raise FileNotFoundError("best_config.json 不存在，請先執行 /backtest/top3")

        best_config = json.loads(config_path.read_text(encoding="utf-8"))
        config_key = "best_overall" if use == "overall" else "best_recent"
        selected_config = best_config.get(config_key)
        if not selected_config:
            raise ValueError("best_config.json 缺少必要欄位。")

        window = int(selected_config["window"])
        if len(recent_draws) < window:
            raise ValueError(f"提供 recent draws 不足 window={window}。")
        train = recent_draws[-window:]

        p_scores, pair_scores = self._build_scores(
            train, alpha=float(selected_config["alpha"])
        )
        method = str(selected_config["method"])
        lam = float(selected_config["lambda"])
        if method == "freq_only":
            top3 = self._predict_hot_only(p_scores)
        elif method == "pair_only":
            top3 = self._predict_pair_only(pair_scores, p_scores)
        else:
            top3 = self._predict_pair_aware(
                p_scores, pair_scores, lam, candidate_pool_size=18
            )

        pair_sum = (
            float(pair_scores[top3[0] - 1, top3[1] - 1])
            + float(pair_scores[top3[0] - 1, top3[2] - 1])
            + float(pair_scores[top3[1] - 1, top3[2] - 1])
        )
        return {
            "top3": list(top3),
            "config_used": {
                "use": use,
                "method": method,
                "window": window,
                "alpha": float(selected_config["alpha"]),
                "lambda": lam,
            },
            "diagnostics": {
                "single_scores": {str(n): float(p_scores[n - 1]) for n in top3},
                "pair_score_sum": pair_sum,
            },
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
        recent_component = self._normalize_vector(
            recent_freq * 0.7 + combo_scores * 0.3
        )

        latest_draw_set = set(recent_draws[-1])
        history_component, similar_cases = self._history_similar_component(
            latest_draw_set, latest_issue
        )
        other_component = self._normalize_vector(zone_weights)

        score = (
            recent_component * weights.recent_weight
            + history_component * weights.history_similar_weight
            + other_component * weights.other_weight
        )

        ranking = np.argsort(score)[::-1] + 1
        selected = ranking[:top_k].tolist()

        # 中文註解：Top3 只代表下一期同一期內三號共現機率最高組合，避免誤解為跨多期分散預測。
        top_triplet = self._best_next_issue_triplet(
            selected,
            score,
            triplet_counter,
            recent_draws,
            weights,
            similar_cases,
            history_component,
            recent_component,
            other_component,
        )
        confidence = self._confidence_labels(score, selected, board_type)
        online_prediction = self.predict_online_top20(recent_draws)

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
            "practical_prediction_top3": online_prediction["top3"],
            "online_strategy": {
                "method": online_prediction["online_method"],
                "predicted_numbers_top20": online_prediction["predicted_numbers_top20"],
                "backtest_summary": online_prediction["backtest_summary"],
            },
            "confidence": confidence,
            "weights": asdict(weights),
            "short_window": short_window,
        }

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

    def _history_similar_component(
        self,
        latest_draw_set: set[int],
        latest_issue: int,
        top_n: int = 100,
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        candidates: List[Tuple[float, int, int]] = []
        for issue, idx in self.issue_to_index.items():
            if issue >= latest_issue:
                continue
            next_issue = issue + 1
            next_idx = self.issue_to_index.get(next_issue)
            if next_idx is None:
                continue
            hist_set = set(self.draw_numbers[idx])
            inter = len(latest_draw_set & hist_set)
            union = len(latest_draw_set | hist_set)
            sim = inter / union if union else 0.0
            candidates.append((sim, issue, next_issue))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = [item for item in candidates if item[0] > 0][:top_n]

        history_scores = np.zeros(80, dtype=float)
        similar_cases: List[Dict[str, object]] = []
        total_weight = sum(sim for sim, _, _ in selected)
        if total_weight <= 0:
            return history_scores, similar_cases

        for sim, issue, next_issue in selected:
            next_draw = self.draw_numbers[self.issue_to_index[next_issue]]
            similar_cases.append(
                {
                    "issue": issue,
                    "next_issue": next_issue,
                    "similarity": round(float(sim), 6),
                }
            )
            for n in next_draw:
                history_scores[n - 1] += sim

        history_scores = history_scores / total_weight
        return history_scores, similar_cases

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
        similar_cases: Sequence[Dict[str, object]],
        history_component: np.ndarray,
        recent_component: np.ndarray,
        other_component: np.ndarray,
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

        similar_case_count = max(len(similar_cases), 1)
        sim_next_counter = Counter()
        for case in similar_cases:
            next_issue = int(case["next_issue"])
            for n in self.draw_numbers[self.issue_to_index[next_issue]]:
                sim_next_counter[n] += 1

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
            recent_strength = float(np.mean([recent_component[n - 1] for n in combo]))
            history_strength = float(np.mean([history_component[n - 1] for n in combo]))
            other_strength = float(np.mean([other_component[n - 1] for n in combo]))
            zone_strength, zone_fit = self._triplet_zone_fit(combo)
            combined_signal = 0.5 * triplet_strength + 0.5 * pair_strength

            combined = (
                weights.recent_weight * (0.6 * recent_strength + 0.4 * combined_signal)
                + weights.history_similar_weight * history_strength
                + weights.other_weight * (0.7 * other_strength + 0.3 * zone_strength)
            )
            if combined > best_score:
                best_score = combined
                best_combo = combo
                number_stats = []
                for n in combo:
                    appear = int(sim_next_counter[n])
                    ratio = appear / similar_case_count
                    number_stats.append(
                        {
                            "number": n,
                            "similar_next_count": appear,
                            "similar_next_ratio": round(float(ratio), 4),
                            "recent_component": round(
                                float(recent_component[n - 1]), 4
                            ),
                            "history_component": round(
                                float(history_component[n - 1]), 4
                            ),
                            "other_component": round(float(other_component[n - 1]), 4),
                            "final_number_score": round(float(score[n - 1]), 4),
                        }
                    )
                best_explain = {
                    "recent_triplet_count": recent_triplet_count,
                    "recent_pair_resonance": round(float(pair_strength), 4),
                    "similar_cases_used": len(similar_cases),
                    "similar_cases_top10": list(similar_cases[:10]),
                    "number_contributions": number_stats,
                    "zone_fit": zone_fit,
                    "weights": asdict(weights),
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


@app.post("/backtest/top3")
def backtest_top3(
    payload: Optional[BacktestRequest] = Body(default=None),
) -> Dict[str, object]:
    analyzer = get_analyzer()
    try:
        return analyzer.run_top3_backtest(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/backtest/hitcount")
def backtest_hitcount(
    payload: Optional[HitCountBacktestRequest] = Body(default=None),
) -> Dict[str, object]:
    analyzer = get_analyzer()
    try:
        return analyzer.run_hitcount_backtest(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/top3")
def predict_top3(
    payload: Optional[PredictRequest] = Body(default=None),
    use: str = Query(default="recent", pattern="^(recent|overall)$"),
) -> Dict[str, object]:
    if payload is None:
        raise HTTPException(status_code=400, detail=PREDICT_REQUIRED_MESSAGE)
    analyzer = get_analyzer()
    recent_draws, _ = _validate_request_recent(payload)
    try:
        return analyzer.predict_top3_with_best(recent_draws=recent_draws, use=use)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

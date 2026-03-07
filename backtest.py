from __future__ import annotations

import json
from pathlib import Path

from agent import BingoAnalyzer


def run_grid_search(
    analyzer: BingoAnalyzer,
    train_window: int = 200,
    max_steps: int = 800,
    output_path: Path = Path("artifacts") / "best_params.json",
) -> dict:
    alphas = [0.7, 0.8, 0.9, 0.95]
    lambdas = [0.3, 0.8, 1.5, 2.5]

    indices = list(range(train_window, len(analyzer.draw_numbers)))[-max_steps:]
    best = None

    for alpha in alphas:
        for lambda_value in lambdas:
            top20_hits = 0
            top10_hits = 0
            top3_hits = 0

            for idx in indices:
                train = analyzer.draw_numbers[idx - train_window : idx]
                actual = set(analyzer.draw_numbers[idx])
                latest_issue = int(analyzer.df.iloc[idx - 1]["issue"])
                pred = analyzer.predict_next(
                    recent_draws=train[-50:],
                    latest_issue=latest_issue,
                    top_k=20,
                    alpha=alpha,
                    lambda_value=lambda_value,
                )
                top20 = set(pred["predicted_numbers_top20"])
                top10 = set(pred["top_10_candidate_numbers"])
                top3 = set(pred["top_10_candidate_numbers"][:3])

                top20_hits += int(len(actual & top20) > 0)
                top10_hits += int(len(actual & top10) > 0)
                top3_hits += int(len(actual & top3) > 0)

            total = max(len(indices), 1)
            metrics = {
                "top20_hit_rate": top20_hits / total,
                "top10_hit_rate": top10_hits / total,
                "top3_hit_rate": top3_hits / total,
            }
            candidate = {
                "alpha": alpha,
                "lambda": lambda_value,
                "metrics": metrics,
            }
            if best is None or (
                metrics["top20_hit_rate"],
                metrics["top10_hit_rate"],
                metrics["top3_hit_rate"],
            ) > (
                best["metrics"]["top20_hit_rate"],
                best["metrics"]["top10_hit_rate"],
                best["metrics"]["top3_hit_rate"],
            ):
                best = candidate

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return best


if __name__ == "__main__":
    analyzer = BingoAnalyzer()
    best = run_grid_search(analyzer)
    print(json.dumps(best, ensure_ascii=False, indent=2))

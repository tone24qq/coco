"""Walk-forward backtest for transformer ranker using same inference contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.build_rank_windows import build_inference_window
from src.history_store import load_local_history
from src.model_transformer import SmallTransformerRanker


def _ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    gains = labels[order]
    denom = np.log2(np.arange(2, k + 2))
    dcg = float((gains / denom).sum())
    ideal = np.sort(labels)[::-1][:k]
    idcg = float((ideal / denom).sum())
    return dcg / idcg if idcg > 0 else 0.0


def run_backtest(
    input_path: Path, model_ckpt: Path, output_dir: Path, window_size: int
) -> None:
    history = load_local_history(input_path)
    model = SmallTransformerRanker.load(model_ckpt)
    model.eval()

    rows: List[Dict[str, object]] = []
    for idx in range(window_size, len(history) - 1):
        context = history.iloc[: idx + 1]
        target = history.iloc[idx + 1]
        window = build_inference_window(context, window_size=window_size)

        x = torch.from_numpy(window.features).unsqueeze(0)
        with torch.no_grad():
            pred = model.predict_scores(x).squeeze(0).cpu().numpy()

        target_numbers = {int(target[f"n{i}"]) for i in range(1, 21)}
        labels = np.array([1.0 if n in target_numbers else 0.0 for n in range(1, 81)])

        top_order = np.argsort(-pred)
        top20 = top_order[:20] + 1
        top10 = top_order[:10] + 1
        top3 = top_order[:3] + 1

        rows.append(
            {
                "issue": str(target["issue"]),
                "hit@20": float(any(int(n) in target_numbers for n in top20)),
                "hit@10": float(any(int(n) in target_numbers for n in top10)),
                "hit@3": float(any(int(n) in target_numbers for n in top3)),
                "ndcg@20": _ndcg_at_k(pred, labels, 20),
                "ndcg@10": _ndcg_at_k(pred, labels, 10),
                "top3_at_least_one_hit": float(
                    any(int(n) in target_numbers for n in top3)
                ),
            }
        )

    if not rows:
        raise ValueError("Insufficient rows for backtest")

    df = pd.DataFrame(rows)
    summary = {
        "hit@20": float(df["hit@20"].mean()),
        "hit@10": float(df["hit@10"].mean()),
        "hit@3": float(df["hit@3"].mean()),
        "ndcg@20": float(df["ndcg@20"].mean()),
        "ndcg@10": float(df["ndcg@10"].mean()),
        "top3_at_least_one_hit": float(df["top3_at_least_one_hit"].mean()),
        "baseline_compare": {"random_hit@20": 20 / 80},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "per_issue_predictions.csv", index=False)
    df.to_json(
        output_dir / "per_issue_predictions.json", orient="records", force_ascii=False
    )
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest transformer ranker")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", default=Path("models/transformer_v1/model.ckpt"), type=Path
    )
    parser.add_argument("--window-size", default=100, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_backtest(args.input, args.model, args.output, args.window_size)


if __name__ == "__main__":
    main()

"""Train transformer ranker with time-series split and ranking objective."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.build_rank_windows import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    TENSOR_CONTRACT,
    build_training_samples,
)
from src.history_store import load_local_history
from src.model_transformer import SmallTransformerRanker, TransformerConfig

MODEL_VERSION = "small_transformer_v2"
MODEL_SIZE_LIMIT_BYTES = 100 * 1024 * 1024


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pos_mask = labels > 0.5
    neg_mask = labels <= 0.5

    losses = []
    for b in range(logits.shape[0]):
        pos = logits[b][pos_mask[b]]
        neg = logits[b][neg_mask[b]]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        losses.append(F.softplus(-diff).mean())

    if not losses:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    return torch.stack(losses).mean()


def _topk_metrics(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    hits = []
    for sample_logit, sample_label in zip(logits, labels):
        top_idx = np.argsort(-sample_logit)[:k]
        hits.append(float(sample_label[top_idx].sum() > 0))
    return float(np.mean(hits))


def _ndcg_at_k(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    scores = []
    for sample_logit, sample_label in zip(logits, labels):
        order = np.argsort(-sample_logit)[:k]
        gains = sample_label[order]
        denom = np.log2(np.arange(2, k + 2))
        dcg = float((gains / denom).sum())

        ideal = np.sort(sample_label)[::-1][:k]
        idcg = float((ideal / denom).sum())
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores))


def _evaluate(
    model: SmallTransformerRanker, loader: DataLoader
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            preds.append(logits.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    metrics = {
        "valid_hit_at_20": _topk_metrics(pred_np, true_np, 20),
        "valid_hit_at_10": _topk_metrics(pred_np, true_np, 10),
        "valid_hit_at_3": _topk_metrics(pred_np, true_np, 3),
        "valid_ndcg_at_20": _ndcg_at_k(pred_np, true_np, 20),
        "valid_top3_at_least_one_hit": _topk_metrics(pred_np, true_np, 3),
    }
    return metrics["valid_ndcg_at_20"], metrics


def train_model(
    input_path: Path,
    output_dir: Path,
    model_file: str,
    window_size: int,
    seed: int,
    epochs: int,
    batch_size: int,
    alpha: float,
    stale_threshold: int,
) -> None:
    _set_seed(seed)

    history = load_local_history(input_path)
    x_all, y_all = build_training_samples(history, window_size=window_size)

    split_idx = int(len(x_all) * 0.8)
    if split_idx <= 0 or split_idx >= len(x_all):
        raise ValueError("Invalid time-series split; need more data")

    x_train = torch.from_numpy(x_all[:split_idx])
    y_train = torch.from_numpy(y_all[:split_idx])
    x_valid = torch.from_numpy(x_all[split_idx:])
    y_valid = torch.from_numpy(y_all[split_idx:])

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False
    )
    valid_loader = DataLoader(
        TensorDataset(x_valid, y_valid), batch_size=batch_size, shuffle=False
    )

    config = TransformerConfig(feature_dim=x_all.shape[-1])
    model = SmallTransformerRanker(config)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    bce = nn.BCEWithLogitsLoss()
    best_metric = -1.0
    best_state = None
    patience, no_improve = 8, 0

    for _epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            bce_loss = bce(logits, y_batch)
            rank_loss = _pairwise_ranking_loss(logits, y_batch)
            loss = bce_loss + alpha * rank_loss
            loss.backward()
            optimizer.step()

        valid_metric, _ = _evaluate(model, valid_loader)
        scheduler.step(valid_metric)

        if valid_metric > best_metric:
            best_metric = valid_metric
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")

    model.load_state_dict(best_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_file
    model.save(model_path)
    model_size = model_path.stat().st_size
    if model_size > MODEL_SIZE_LIMIT_BYTES:
        raise ValueError(
            f"Model artifact too large: {model_size} exceeds {MODEL_SIZE_LIMIT_BYTES}"
        )

    _, metrics = _evaluate(model, valid_loader)
    trained_up_to_issue = str(history.iloc[split_idx + window_size - 1]["issue"])

    metadata = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "tensor_contract": TENSOR_CONTRACT,
        "model_file": model_file,
        "window_size": window_size,
        "seed": seed,
        "train_samples": int(len(x_train)),
        "valid_samples": int(len(x_valid)),
        "trained_up_to_issue": trained_up_to_issue,
        "stale_threshold": stale_threshold,
        "baseline_metrics": metrics,
        "expected_input_schema": [
            "issue",
            "draw_time",
            *[f"n{i}" for i in range(1, 21)],
        ],
        "expected_output_schema": [
            "latest_known_issue",
            "target_issue",
            "model_version",
            "feature_version",
            "data_source",
            "scores",
            "top20",
            "top3",
        ],
    }
    (output_dir / "transformer_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer ranker")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-file", default="model.ckpt")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--stale-threshold", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        input_path=args.input,
        output_dir=args.output,
        model_file=args.model_file,
        window_size=args.window_size,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        stale_threshold=args.stale_threshold,
    )


if __name__ == "__main__":
    main()

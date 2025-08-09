from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from src.config import load_config
from src.models.maskgit import MaskGit
from src.training.datasets import JsonBoardsDataset, pad_collate
from src.training.loss_vec import compute_loss_vectorized
from src.training.train import evaluate, set_seed


def find_datasets(
    data_dir: Path = Path(__file__).parent / "src" / "data",
) -> List[Tuple[int, int, Path]]:
    """Find dataset JSON files like ``4x5.json`` under ``data_dir``."""

    pattern = re.compile(r"^(\d+)x(\d+)\.json$")
    out: List[Tuple[int, int, Path]] = []
    for p in data_dir.glob("*.json"):
        m = pattern.match(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    return out


def train_one(
    path: Path,
    rows: int,
    cols: int,
    *,
    epochs: int = 10,
    bsz: int = 32,
    lr: float = 3e-4,
    seed: int = 42,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[資料] 使用 {path} 訓練 {rows}x{cols} 模型")
    set_seed(seed)
    cfg = load_config("configs/train_small_mix.yaml")
    model = MaskGit(
        cfg["vocab_size"],
        cfg["model"]["d_model"],
        cfg["model"]["n_head"],
        cfg["model"]["num_layers"],
    ).to(device)

    dataset = JsonBoardsDataset(path, mask_target=True, seed=seed)
    if len(dataset) < 2:
        raise ValueError("dataset 太小，至少需 2 筆資料")
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=bsz,
        shuffle=True,
        num_workers=2,
        collate_fn=pad_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bsz,
        shuffle=False,
        num_workers=2,
        collate_fn=pad_collate,
        pin_memory=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    outdir = Path("weights")
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / f"{rows}x{cols}.ckpt"

    for epoch in range(1, epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader, 1):
            tokens = batch["tokens"].to(device)
            target = batch["target"].to(device)
            attn = batch["attn_mask"].to(device)
            N = batch["N"].to(device)

            logits = model(tokens, attn)
            loss = compute_loss_vectorized(logits, target, N)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                print(
                    f"[訓練] {rows}x{cols} 第 {epoch} 代 第 {step} 步 損失={loss.item():.4f}"
                )

        val = evaluate(model, val_loader, device)
        print(f"[驗證] {rows}x{cols} 第 {epoch} 代 損失={val['loss']:.6f}")
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"[保存] {rows}x{cols} 新最佳模型 -> {ckpt_path}")


def main() -> None:  # pragma: no cover - CLI
    datasets = find_datasets()
    if not datasets:
        print("[警告] 未找到任何資料集，請將 NxY.json 放於 src/data 目錄下")
        return
    for rows, cols, path in datasets:
        print(f"[開始] 訓練 {rows}x{cols} 模型")
        train_one(path, rows, cols)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

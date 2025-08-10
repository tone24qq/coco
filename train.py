from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from src.config import load_config
from src.models.maskgit import MaskGit
from src.training.datasets import JsonBoardsDataset, pad_collate
from src.training.dep_bias import apply_dep_bias
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
    target_loss: float | None = None,  # 門檻：達標後「等到 epoch 結束」才早停
    use_dep_bias: bool = False,
    dep_alpha: float = 0.5,
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

    # 若你的資料包含 target 欄位，用 mask_target=True 會只遮住目標；
    # 你也可以改 JsonBoardsDataset 的參數以符合當前資料版型。
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
        hit_threshold_this_epoch = False  # 做法B：本 epoch 內是否曾達標

        for step, batch in enumerate(train_loader, 1):
            tokens = batch["tokens"].to(device)
            target = batch["target"].to(device)
            attn = batch["attn_mask"].to(device)
            N = batch["N"].to(device)
            rows_t = batch["rows"].to(device)
            cols_t = batch["cols"].to(device)

            logits = model(tokens, attn)
            if use_dep_bias:
                apply_dep_bias(logits, tokens, target, rows_t, cols_t, N, dep_alpha)
            loss = compute_loss_vectorized(logits, target, N)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                print(
                    f"[訓練] {rows}x{cols} 第 {epoch} 代 第 {step} 步 損失={loss.item():.4f}"
                )

            # 標記達標，但不 break —— 等 epoch 結束後再處理
            if target_loss is not None and loss.item() <= target_loss:
                hit_threshold_this_epoch = True

        # ---- 一個 epoch 結束後才評估與可能早停 ----
        val = evaluate(
            model,
            val_loader,
            device,
            use_dep_bias=use_dep_bias,
            dep_alpha=dep_alpha,
        )
        print(f"[驗證] {rows}x{cols} 第 {epoch} 代 損失={val['loss']:.6f}")
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"[保存] {rows}x{cols} 新最佳模型 -> {ckpt_path}")

        if hit_threshold_this_epoch:
            print(
                f"[早停] 第 {epoch} 代內曾達到門檻 {target_loss:.6g}，於 epoch 結束觸發停止。"
            )
            # 再存一次，確保有最後權重（若上面已是最佳也OK）
            torch.save(model.state_dict(), ckpt_path)
            print(f"[保存] {rows}x{cols} 收斂模型 -> {ckpt_path}")
            break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="src/data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bsz", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu；預設自動偵測")
    p.add_argument(
        "--target_loss",
        type=float,
        default=None,
        help="達到此訓練 loss 門檻時，不中斷步數；等該 epoch 結束後早停",
    )
    p.add_argument(
        "--use_dep_bias",
        action="store_true",
        help="計算同行+同列直方圖先驗並加到被遮蔽位置的 logits",
    )
    p.add_argument(
        "--dep_alpha",
        type=float,
        default=0.5,
        help="依賴偏置權重（加入 logits 前的縮放係數）",
    )
    return p.parse_args()


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    datasets = find_datasets(Path(args.data_dir))
    if not datasets:
        print("[警告] 未找到任何資料集，請將 NxY.json 放於 src/data 目錄下")
        return
    for rows, cols, path in datasets:
        print(f"[開始] 訓練 {rows}x{cols} 模型")
        train_one(
            path,
            rows,
            cols,
            epochs=args.epochs,
            bsz=args.bsz,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            target_loss=args.target_loss,
            use_dep_bias=args.use_dep_bias,
            dep_alpha=args.dep_alpha,
        )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

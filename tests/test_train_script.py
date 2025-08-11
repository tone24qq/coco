import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import train
from train import iterative_fill, load_dataset_dir, train_one_bucket


def test_load_dataset_dir(tmp_path: Path):
    board = [[1, 2], [3, 4]]
    json_path = tmp_path / "b.json"
    json_path.write_text(json.dumps({"boards": [{"grid": board}]}))
    jsonl_path = tmp_path / "b.jsonl"
    jsonl_path.write_text(json.dumps({"grid": board}) + "\n")
    buckets = load_dataset_dir(str(tmp_path))
    assert (2, 2) in buckets
    assert len(buckets[(2, 2)]) == 2


def test_iterative_fill_uniqueness():
    class Dummy(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits

        def forward(self, tokens):  # type: ignore[override]
            return self.logits

    tokens = torch.tensor([[1, -1, -1, -1]], dtype=torch.long)
    logits = torch.ones(1, 4, 4)
    logits[0, :, 0] = 5  # prefer number 1 everywhere
    dummy = Dummy(logits)
    out = iterative_fill(
        dummy, tokens.clone(), V=4, steps=4, tau_start=0.0, tau_end=0.0, per_step_k=1
    )
    vals = out[0].tolist()
    assert vals.count(1) == 1
    assert sorted(vals) == [1, 2, 3, 4]


def test_train_one_bucket_runs(tmp_path: Path):
    boards = [np.array([[1, 2], [3, 4]], dtype=np.int64) for _ in range(5)]
    args = SimpleNamespace(
        mask_lo=0.2,
        mask_hi=0.5,
        bsz=2,
        d_model=32,
        nhead=4,
        depth=1,
        dim_ff=64,
        dropout=0.0,
        lr=1e-3,
        wd=0.0,
        cpu=True,
        val_ratio=0.2,
        uniq_lambda=0.5,
        uniq_soft_penalty=False,
        grad_clip=1.0,
        eval_mask_rate=0.5,
        decode_steps=2,
        fail_buffer_max=10,
        epochs=1,
        out_dir=str(tmp_path),
    )
    best = train_one_bucket((2, 2), boards, args)
    assert 0.0 <= best <= 1.0
    ckpt = Path(args.out_dir) / "checkpoints/2x2/best.pth"
    assert ckpt.exists()


def test_train_one_bucket_early_stops(tmp_path: Path, monkeypatch):
    boards = [np.array([[1, 2], [3, 4]], dtype=np.int64) for _ in range(5)]
    args = SimpleNamespace(
        mask_lo=0.2,
        mask_hi=0.5,
        bsz=2,
        d_model=32,
        nhead=4,
        depth=1,
        dim_ff=64,
        dropout=0.0,
        lr=1e-3,
        wd=0.0,
        cpu=True,
        val_ratio=0.2,
        uniq_lambda=0.5,
        uniq_soft_penalty=False,
        grad_clip=1.0,
        eval_mask_rate=0.5,
        decode_steps=2,
        fail_buffer_max=10,
        epochs=5,
        out_dir=str(tmp_path),
    )
    calls = {"n": 0}

    def fake_emr(model, boards, mask_rate=0.5):  # pragma: no cover - testing hook
        calls["n"] += 1
        return 0.0

    monkeypatch.setattr(train, "exact_match_rate", fake_emr)
    train_one_bucket((2, 2), boards, args)
    assert calls["n"] == 2

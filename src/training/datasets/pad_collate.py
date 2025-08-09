from __future__ import annotations

import torch


def pad_collate(samples):
    """Pad variable-length samples into a batch.

    Each sample is a dict with keys ``tokens``, ``target``, ``attn_mask`` and
    metadata ``rows``, ``cols`` and ``N``. ``tokens`` and ``target`` are padded
    with zeros to the maximum length in the batch. ``attn_mask`` is padded with
    ``False``. ``rows``, ``cols`` and ``N`` are stacked without padding.
    """

    max_len = max(s["tokens"].numel() for s in samples)
    batch = {}

    for key in ("tokens", "target", "attn_mask"):
        padded = []
        for s in samples:
            t = s[key]
            if t.numel() < max_len:
                pad = torch.zeros(max_len - t.numel(), dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        batch[key] = torch.stack(padded, dim=0)

    for key in ("rows", "cols", "N"):
        batch[key] = torch.tensor([s[key] for s in samples], dtype=torch.long)

    return batch

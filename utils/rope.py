from __future__ import annotations

from typing import Tuple

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def build_rope_cache(
    seq_len: int, dim: int, base: float = 10_000.0, device: torch.device | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    建立 RoPE 需要的 cos/sin 快取。
    dim 需為偶數（對偶維度旋轉），若為奇數會自動補到偶數最後一維不旋轉。
    回傳張量 shape: (seq_len, dim)
    """
    half = dim // 2
    theta = torch.arange(half, device=device, dtype=torch.float32)
    theta = 1.0 / (base ** (theta / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    angles = pos * theta.unsqueeze(0)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    if dim % 2 == 1:
        cos = torch.nn.functional.pad(cos, (0, 1))
        sin = torch.nn.functional.pad(sin, (0, 1))
    return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """對 q,k 套用 RoPE"""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot_even = -x_odd
        x_rot_odd = x_even
        out = torch.empty_like(x)
        out[..., ::2] = x_rot_even
        out[..., 1::2] = x_rot_odd
        if x.shape[-1] % 2 == 1:
            out[..., -1] = 0
        return out

    q_rot = _rotate(q)
    k_rot = _rotate(k)
    q_out = q * cos + q_rot * sin
    k_out = k * cos + k_rot * sin
    return q_out, k_out

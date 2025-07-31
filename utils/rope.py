from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)
_rope2d_logged = False

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
    freq = torch.arange(half, device=device, dtype=torch.float32)
    freq = 1.0 / (base ** (freq / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    angles = pos * freq.unsqueeze(0)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.stack((cos, cos), dim=-1).reshape(seq_len, half * 2)
    sin = torch.stack((sin, sin), dim=-1).reshape(seq_len, half * 2)
    if dim % 2 == 1:  # pad 1 col for odd dim
        cos = torch.nn.functional.pad(cos, (0, 1))
        sin = torch.nn.functional.pad(sin, (0, 1))
    return cos[:, :dim], sin[:, :dim]


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 1D RoPE to ``q`` and ``k``."""
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


def apply_rope_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_row: torch.Tensor,
    sin_row: torch.Tensor,
    cos_col: torch.Tensor,
    sin_col: torch.Tensor,
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D RoPE using row and column caches."""

    global _rope2d_logged
    if logger.isEnabledFor(logging.DEBUG) and not _rope2d_logged:
        logger.debug("已套用二維相對位置編碼，張量形狀：%s", tuple(q.shape))
        _rope2d_logged = True

    row_dim = q.shape[-1] // 2

    q_row, q_col = q[..., :row_dim], q[..., row_dim:]
    k_row, k_col = k[..., :row_dim], k[..., row_dim:]

    row_cos = cos_row[row_ids]
    row_sin = sin_row[row_ids]
    col_cos = cos_col[col_ids]
    col_sin = sin_col[col_ids]

    q_row, k_row = apply_rope(q_row, k_row, row_cos, row_sin)
    q_col, k_col = apply_rope(q_col, k_col, col_cos, col_sin)

    q_out = torch.cat([q_row, q_col], dim=-1)
    k_out = torch.cat([k_row, k_col], dim=-1)
    return q_out, k_out

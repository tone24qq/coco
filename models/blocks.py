from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[misc]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from flash_attn import flash_attn_func  # type: ignore

    HAS_FLASH = True
except Exception:  # pragma: no cover
    HAS_FLASH = False

from utils.rope import apply_rope_2d, build_rope_cache

logger = logging.getLogger(__name__)

if not TORCH_AVAILABLE:

    class RMSNorm:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("torch is required for RMSNorm")

    class SwiGLU:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("torch is required for SwiGLU")

    @dataclass
    class AttnConfig:
        dim: int
        n_heads: int
        dropout: float = 0.0
        rope_base: float = 10_000.0
        use_flash: bool = False

    class RoPEAttention:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("torch is required for RoPEAttention")

    class TransformerBlock:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("torch is required for TransformerBlock")

else:

    class RMSNorm(nn.Module):
        """RMSNorm，較 LayerNorm 穩定、便宜。"""

        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return self.weight * x * rms

    class SwiGLU(nn.Module):
        """SwiGLU 前饋，提升表示力與穩定性。"""

        def __init__(self, dim: int, hidden_mult: float = 2.0, dropout: float = 0.0):
            super().__init__()
            hidden = int(dim * hidden_mult)
            self.w1 = nn.Linear(dim, hidden, bias=False)
            self.w2 = nn.Linear(dim, hidden, bias=False)
            self.proj = nn.Linear(hidden, dim, bias=False)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            a = self.w1(x)
            b = self.w2(x)
            x = F.silu(a) * b
            x = self.proj(x)
            return self.dropout(x)

    @dataclass
    class AttnConfig:
        dim: int
        n_heads: int
        dropout: float = 0.0
        rope_base: float = 10_000.0
        use_flash: bool = False

    class RoPEAttention(nn.Module):
        """Multi-head attention with optional 2D RoPE."""

        def __init__(self, cfg: AttnConfig, *, rows: int = 1, cols: int | None = None):
            super().__init__()
            assert cfg.dim % cfg.n_heads == 0, "dim must be divisible by n_heads"
            self.cfg = cfg
            self.head_dim = cfg.dim // cfg.n_heads
            self.qkv = nn.Linear(cfg.dim, cfg.dim * 3, bias=False)
            self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.attn_dropout = nn.Dropout(cfg.dropout)
            self.resid_dropout = nn.Dropout(cfg.dropout)
            self.register_buffer("_cos", torch.empty(0), persistent=False)
            self.register_buffer("_sin", torch.empty(0), persistent=False)
            self.register_buffer("_cos_row", torch.empty(0), persistent=False)
            self.register_buffer("_sin_row", torch.empty(0), persistent=False)
            self.register_buffer("_cos_col", torch.empty(0), persistent=False)
            self.register_buffer("_sin_col", torch.empty(0), persistent=False)
            self.rows = int(rows)
            self.cols = int(cols if cols is not None else rows)
            if self.rows * self.cols <= 0:
                raise ValueError("rows and cols must be positive")

        def _rope_cache_2d(
            self, device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            row_dim = self.head_dim // 2
            col_dim = self.head_dim - row_dim
            if (
                self._cos_row.numel() == 0
                or self._cos_row.size(0) < self.rows
                or self._cos_row.device != device
            ):
                cos_r, sin_r = build_rope_cache(
                    self.rows, row_dim, base=self.cfg.rope_base, device=device
                )
                self._cos_row = cos_r
                self._sin_row = sin_r
            if (
                self._cos_col.numel() == 0
                or self._cos_col.size(0) < self.cols
                or self._cos_col.device != device
            ):
                cos_c, sin_c = build_rope_cache(
                    self.cols, col_dim, base=self.cfg.rope_base, device=device
                )
                self._cos_col = cos_c
                self._sin_col = sin_c
            return (
                self._cos_row[: self.rows],
                self._sin_row[: self.rows],
                self._cos_col[: self.cols],
                self._sin_col[: self.cols],
            )

        def forward(
            self,
            x: torch.Tensor,
            row_ids: torch.Tensor,
            col_ids: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            B, N, D = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            H = self.cfg.n_heads
            q = q.view(B, N, H, self.head_dim).transpose(1, 2)
            k = k.view(B, N, H, self.head_dim).transpose(1, 2)
            v = v.view(B, N, H, self.head_dim).transpose(1, 2)

            cos_r, sin_r, cos_c, sin_c = self._rope_cache_2d(x.device)
            q, k = apply_rope_2d(q, k, cos_r, sin_r, cos_c, sin_c, row_ids, col_ids)
            if logger.isEnabledFor(logging.DEBUG):
                dr = row_ids[1:] - row_ids[:-1]
                dc = col_ids[1:] - col_ids[:-1]
                logger.debug(
                    "q/k 已旋轉，行(row)變動均值=%.2f，列(col)變動均值=%.2f",
                    dr.float().abs().mean().item(),
                    dc.float().abs().mean().item(),
                )

            if self.cfg.use_flash and HAS_FLASH and x.is_cuda:
                q_f = q.transpose(1, 2)
                k_f = k.transpose(1, 2)
                v_f = v.transpose(1, 2)
                out = flash_attn_func(
                    q_f,
                    k_f,
                    v_f,
                    dropout_p=self.cfg.dropout if self.training else 0.0,
                    softmax_scale=None,
                    causal=False,
                )
                out = out.transpose(1, 2).contiguous()
            else:
                att = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
                if attn_mask is not None:
                    att = att.masked_fill(attn_mask == 0, float("-inf"))
                att = torch.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                out = torch.matmul(att, v)

            out = out.transpose(1, 2).contiguous().view(B, N, D)
            out = self.out(out)
            return self.resid_dropout(out)

    class TransformerBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            n_heads: int,
            dropout: float = 0.0,
            hidden_mult: float = 2.0,
            use_flash: bool = False,
            *,
            rows: int = 1,
            cols: int | None = None,
            attn_module: Optional[nn.Module] = None,
        ) -> None:
            """Initialize transformer block.

            Parameters
            ----------
            dim:
                Model dimensionality.
            n_heads:
                Number of attention heads.
            dropout:
                Dropout rate.
            hidden_mult:
                Expansion factor for the feedforward network.
            use_flash:
                Whether to enable flash attention for the default attention
                module.
            rows, cols:
                Grid shape for positional encodings.
            attn_module:
                Optional custom attention module. When provided, this module is
                used instead of the default :class:`RoPEAttention`.
            """

            super().__init__()
            self.norm1 = RMSNorm(dim)
            if attn_module is None:
                self.attn = RoPEAttention(
                    AttnConfig(
                        dim=dim,
                        n_heads=n_heads,
                        dropout=dropout,
                        use_flash=use_flash,
                    ),
                    rows=rows,
                    cols=cols,
                )
            else:
                self.attn = attn_module
            self.norm2 = RMSNorm(dim)
            self.ff = SwiGLU(dim, hidden_mult=hidden_mult, dropout=dropout)

        def forward(
            self,
            x: torch.Tensor,
            row_ids: torch.Tensor,
            col_ids: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:  # noqa: D401
            x = x + self.attn(self.norm1(x), row_ids, col_ids, attn_mask)
            x = x + self.ff(self.norm2(x))
            return x

from __future__ import annotations

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

from utils.rope import apply_rope, build_rope_cache

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
        """多頭注意力，內建 RoPE，可選 FlashAttention-2。"""

        def __init__(self, cfg: AttnConfig):
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

        def _rope_cache(
            self, seq_len: int, device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if (
                self._cos.numel() == 0
                or self._cos.size(0) < seq_len
                or self._cos.device != device
            ):
                cos, sin = build_rope_cache(
                    seq_len, self.head_dim, base=self.cfg.rope_base, device=device
                )
                self._cos = cos
                self._sin = sin
            return self._cos[:seq_len], self._sin[:seq_len]

        def forward(
            self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            B, N, D = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            H = self.cfg.n_heads
            q = q.view(B, N, H, self.head_dim).transpose(1, 2)
            k = k.view(B, N, H, self.head_dim).transpose(1, 2)
            v = v.view(B, N, H, self.head_dim).transpose(1, 2)

            cos, sin = self._rope_cache(N, x.device)
            q, k = apply_rope(q, k, cos, sin)

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
        ):
            super().__init__()
            self.norm1 = RMSNorm(dim)
            self.attn = RoPEAttention(
                AttnConfig(
                    dim=dim, n_heads=n_heads, dropout=dropout, use_flash=use_flash
                )
            )
            self.norm2 = RMSNorm(dim)
            self.ff = SwiGLU(dim, hidden_mult=hidden_mult, dropout=dropout)

        def forward(
            self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:  # noqa: D401
            x = x + self.attn(self.norm1(x), attn_mask)
            x = x + self.ff(self.norm2(x))
            return x

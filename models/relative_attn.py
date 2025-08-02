try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[misc]
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class Relative2DAttention(nn.Module):
        """Multi-head attention with learnable 2D relative position bias."""

        def __init__(
            self,
            d_model: int,
            nhead: int,
            max_rel_row: int,
            max_rel_col: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.nhead = nhead
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.row_bias = nn.Embedding(2 * max_rel_row + 1, nhead)
            self.col_bias = nn.Embedding(2 * max_rel_col + 1, nhead)
            self.max_rel_row = max_rel_row
            self.max_rel_col = max_rel_col

        def forward(
            self,
            x: torch.Tensor,
            row_ids: torch.Tensor,
            col_ids: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Apply attention with relative 2D bias."""

            B, L, D = x.shape
            x2 = x.transpose(0, 1)  # (L, B, D)

            r_i = row_ids[:, None]
            r_j = row_ids[None, :]
            rel_row = (r_j - r_i).clamp(
                -self.max_rel_row, self.max_rel_row
            ) + self.max_rel_row

            c_i = col_ids[:, None]
            c_j = col_ids[None, :]
            rel_col = (c_j - c_i).clamp(
                -self.max_rel_col, self.max_rel_col
            ) + self.max_rel_col

            bias = self.row_bias(rel_row) + self.col_bias(rel_col)
            bias = bias.permute(2, 0, 1).unsqueeze(1).repeat(1, B, 1, 1)
            bias = bias.reshape(self.nhead * B, L, L).to(x.dtype)

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).expand_as(bias)
                else:
                    attn_mask = attn_mask.to(x.dtype)
                bias = bias + attn_mask

            out, _ = self.attn(x2, x2, x2, attn_mask=bias)
            return out.transpose(0, 1)

else:  # pragma: no cover - torch missing

    class Relative2DAttention:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise RuntimeError("torch is required for Relative2DAttention")

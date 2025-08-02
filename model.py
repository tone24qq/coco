try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[misc]
    TORCH_AVAILABLE = False

from models.blocks import TransformerBlock


class DynamicMET(nn.Module if TORCH_AVAILABLE else object):
    """Simple transformer-like model for tabular scratch card prediction."""

    def __init__(
        self,
        num_fields: int,
        num_values: int = 81,
        d_model: int = 128,
        nhead: int = 4,
        depth: int = 6,
        dropout: float = 0.0,
        use_flash: bool = False,
        *,
        rows: int = 1,
        cols: int | None = None,
    ) -> None:
        if num_values != 81:
            raise ValueError("num_values must be 81 (including blank)")
        self.num_fields = int(num_fields)
        self.num_values = int(num_values)
        self.d_model = int(d_model)
        self.rows = int(rows)
        self.cols = int(cols if cols is not None else rows if rows > 0 else 1)
        if self.rows * self.cols != self.num_fields:
            raise ValueError("rows*cols must equal num_fields")
        if TORCH_AVAILABLE:
            super().__init__()  # type: ignore[misc]
            self.token_emb = nn.Embedding(num_values, d_model)
            row_dim = d_model // 2
            col_dim = d_model - row_dim
            self.row_emb = nn.Embedding(self.rows, row_dim)
            self.col_emb = nn.Embedding(self.cols, col_dim)
            self.embed_dropout = nn.Dropout(dropout)
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model,
                        nhead,
                        dropout=dropout,
                        hidden_mult=2.0,
                        use_flash=use_flash,
                        rows=self.rows,
                        cols=self.cols,
                    )
                    for _ in range(depth)
                ]
            )
            self.norm_out = nn.LayerNorm(d_model)
            self.classifier = nn.Linear(d_model, num_values)
            assert self.classifier.out_features == 81, "num_values 必須是 81 (含空白)"
        else:
            self.token_emb = None
            self.row_emb = None
            self.col_emb = None
            self.embed_dropout = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        if TORCH_AVAILABLE:
            _, N = x.shape
            tok = self.token_emb(x)
            pos_ids = torch.arange(N, device=x.device)
            row_ids = torch.div(pos_ids, self.cols, rounding_mode="floor")
            col_ids = pos_ids % self.cols
            pos = torch.cat([self.row_emb(row_ids), self.col_emb(col_ids)], dim=-1)
            tok = tok + pos.unsqueeze(0)
            tok = self.embed_dropout(tok)
            h = tok
            for blk in self.blocks:
                h = blk(h, row_ids, col_ids, attn_mask=None)
            h = self.norm_out(h)
            logits = self.classifier(h)
            return logits
        import numpy as np

        bsz, fields = x.shape
        return np.zeros((bsz, fields, self.num_values), dtype=float)

    __call__ = forward

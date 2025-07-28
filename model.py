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
        num_values: int,
        d_model: int = 128,
        nhead: int = 4,
        depth: int = 6,
        dropout: float = 0.0,
        use_flash: bool = False,
    ) -> None:
        self.num_fields = int(num_fields)
        self.num_values = int(num_values)
        self.d_model = int(d_model)
        if TORCH_AVAILABLE:
            super().__init__()  # type: ignore[misc]
            self.token_emb = nn.Embedding(num_values + 1, d_model)
            self.field_embed = nn.Embedding(self.num_fields, d_model)
            self.embed_dropout = nn.Dropout(dropout)
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model,
                        nhead,
                        dropout=dropout,
                        hidden_mult=2.0,
                        use_flash=use_flash,
                    )
                    for _ in range(depth)
                ]
            )
            self.norm_out = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_values + 1)
        else:
            self.token_emb = None
            self.field_embed = None
            self.embed_dropout = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        if TORCH_AVAILABLE:
            _, N = x.shape
            tok = self.token_emb(x)
            pos_ids = torch.arange(N, device=x.device)
            pos = self.field_embed(pos_ids)
            tok = tok + pos.unsqueeze(0)
            tok = self.embed_dropout(tok)
            h = tok
            for blk in self.blocks:
                h = blk(h, attn_mask=None)
            h = self.norm_out(h)
            logits = self.head(h)
            return logits
        import numpy as np

        bsz, fields = x.shape
        return np.zeros((bsz, fields, self.num_values + 1), dtype=float)

    __call__ = forward

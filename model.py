try:
    import torch
    import torch.nn as nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None
    nn = None  # type: ignore
    TransformerEncoder = TransformerEncoderLayer = None  # type: ignore
    TORCH_AVAILABLE = False


class DynamicMET(nn.Module if TORCH_AVAILABLE else object):
    """Dynamic Masked Encoding Transformer for scratch-card boards.

    類別索引規則：
    - 0 = MASK（保留，不作為實際數字）
    - 1..N = 盤面實際數字（唯一）
    因此輸出維度為 N+1，訓緻時使用 ``ignore_index=0``。
    """

    def __init__(
        self,
        num_fields: int,
        num_values: int,
        d_model: int = 128,
        nhead: int = 4,
        depth: int = 6,
    ) -> None:
        # num_values = N (R*C)
        # logits dimension is N+1 with class 0 reserved for MASK
        self.num_fields = num_fields
        self.num_values = num_values
        if TORCH_AVAILABLE:
            super().__init__()  # type: ignore[misc]
            # Embeddings for values and positions
            self.value_embed = nn.Embedding(num_values + 1, d_model)
            self.field_embed = nn.Embedding(num_fields, d_model)
            # Use batch_first to support nested tensor optimization
            encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True)
            # Enable nested tensor for potential speed and memory benefits
            self.transformer = TransformerEncoder(
                encoder_layer, num_layers=depth, enable_nested_tensor=True
            )
            self.head = nn.Linear(d_model, num_values + 1)

    def forward(self, input_vals):  # type: ignore[override]
        """Forward pass returning logits for each position."""
        if TORCH_AVAILABLE:
            assert isinstance(input_vals, torch.Tensor)
            bsz, fields = input_vals.shape
            # value embedding + positional embedding
            x = self.value_embed(input_vals)
            pos = self.field_embed(torch.arange(fields, device=x.device))
            x = x + pos.unsqueeze(0)
            # Transformer expects (batch, seq, dim) with batch_first
            z = self.transformer(x)
            return self.head(z)
        import numpy as np

        # Fallback: return zeros
        bsz, fields = input_vals.shape
        return np.zeros((bsz, fields, self.num_values + 1), dtype=float)

    __call__ = forward

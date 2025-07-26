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

    If ``torch`` is unavailable, operations fall back to NumPy and return
    zeros so that dependent services remain functional.
    """

    def __init__(
        self,
        num_fields: int,
        num_values: int,
        d_model: int = 128,
        nhead: int = 4,
        depth: int = 6,
    ) -> None:
        self.num_fields = num_fields
        self.num_values = num_values
        if TORCH_AVAILABLE:
            super().__init__()  # type: ignore[misc]
            self.value_embed = nn.Embedding(num_values + 1, d_model)
            self.field_embed = nn.Embedding(num_fields, d_model)
            encoder_layer = TransformerEncoderLayer(d_model, nhead)
            self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
            self.head = nn.Linear(d_model, num_values)

    def __call__(self, input_vals):
        """Return logits for each position with optional NumPy fallback."""
        if TORCH_AVAILABLE:
            assert isinstance(input_vals, torch.Tensor)
            bsz, fields = input_vals.shape
            x = self.value_embed(input_vals)
            pos = self.field_embed(torch.arange(fields, device=x.device))
            x = x + pos.unsqueeze(0)
            x = x.permute(1, 0, 2)
            z = self.transformer(x)
            z = z.permute(1, 0, 2)
            return self.head(z)
        import numpy as np

        bsz, fields = input_vals.shape
        return np.zeros((bsz, fields, self.num_values), dtype=float)

    # retain PyTorch-style API when torch is missing
    def eval(self):  # type: ignore[override]
        return self

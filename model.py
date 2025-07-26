import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DynamicMET(nn.Module):
    """Dynamic Masked Encoding Transformer for scratch-card boards."""

    def __init__(
        self,
        num_fields: int,
        num_values: int,
        d_model: int = 128,
        nhead: int = 4,
        depth: int = 6,
    ) -> None:
        super().__init__()
        self.value_embed = nn.Embedding(num_values + 1, d_model)
        self.field_embed = nn.Embedding(num_fields, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(d_model, num_values)

    def forward(self, input_vals: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for each position."""
        bsz, fields = input_vals.shape
        x = self.value_embed(input_vals)
        pos = self.field_embed(torch.arange(fields, device=x.device))
        x = x + pos.unsqueeze(0)
        x = x.permute(1, 0, 2)
        z = self.transformer(x)
        z = z.permute(1, 0, 2)
        return self.head(z)

"""PyTorch Small Transformer ranker for candidate ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn


@dataclass
class TransformerConfig:
    layers: int = 3
    d_model: int = 128
    nhead: int = 8
    dim_feedforward: int = 256
    dropout: float = 0.1
    feature_dim: int = 24
    max_candidates: int = 80


class SmallTransformerRanker(nn.Module):
    """Tensor contract:
    - raw tensor: [batch, 80, feature_dim]
    - model input tensor: [batch, 80, d_model]
    - attention axis: candidate-to-candidate self-attention
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.feature_dim, config.d_model)
        self.number_embedding = nn.Embedding(config.max_candidates + 1, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
        self.head = nn.Linear(config.d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input [batch,80,feature_dim], got ndim={x.ndim}"
            )
        if x.shape[1] != self.config.max_candidates:
            raise ValueError(f"Expected candidate axis=80, got {x.shape[1]}")
        if x.shape[2] != self.config.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.config.feature_dim}, got {x.shape[2]}"
            )

        batch_size = x.shape[0]
        number_ids = torch.arange(1, self.config.max_candidates + 1, device=x.device)
        number_ids = number_ids.unsqueeze(0).expand(batch_size, -1)

        hidden = self.input_proj(x) + self.number_embedding(number_ids)
        hidden = self.input_norm(hidden)
        hidden = self.input_dropout(hidden)

        encoded = self.encoder(hidden)
        scores = self.head(encoded).squeeze(-1)
        return scores

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def save(self, path: Path | str, metadata: Dict[str, Any] | None = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "config": asdict(self.config),
            "metadata": metadata or {},
        }
        torch.save(payload, str(path))

    @classmethod
    def load(
        cls, path: Path | str, config: TransformerConfig | None = None
    ) -> "SmallTransformerRanker":
        payload = torch.load(str(path), map_location="cpu")  # nosec B614
        saved_cfg = TransformerConfig(**payload["config"])
        runtime_cfg = config or saved_cfg
        if asdict(runtime_cfg) != asdict(saved_cfg):
            raise ValueError(
                "Tensor contract mismatch between runtime config and checkpoint"
            )

        model = cls(runtime_cfg)
        model.load_state_dict(payload["state_dict"])
        return model

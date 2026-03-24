from pathlib import Path

import pytest
import torch

from src.model_transformer import SmallTransformerRanker, TransformerConfig


def test_model_forward_tensor_shape() -> None:
    model = SmallTransformerRanker(TransformerConfig(feature_dim=24))
    x = torch.randn(2, 80, 24)
    y = model(x)
    if tuple(y.shape) != (2, 80):
        pytest.fail("model output shape mismatch")


def test_model_save_load_consistency(tmp_path: Path) -> None:
    config = TransformerConfig(feature_dim=24)
    model = SmallTransformerRanker(config)
    x = torch.randn(1, 80, 24)
    out1 = model.predict_scores(x)

    ckpt = tmp_path / "model.ckpt"
    model.save(ckpt)
    loaded = SmallTransformerRanker.load(ckpt, config)
    out2 = loaded.predict_scores(x)

    if not torch.allclose(out1, out2, atol=1e-6):
        pytest.fail("save/load prediction mismatch")

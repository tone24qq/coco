import pytest

from model import DynamicMET
from models.relative_attn import Relative2DAttention

torch = pytest.importorskip("torch")


def test_dynamic_met_forward() -> None:
    num_fields = 20
    model = DynamicMET(num_fields=num_fields, num_values=num_fields, rows=4, cols=5)
    x = torch.randint(0, model.num_values, (2, num_fields))
    out = model(x)
    assert out.shape == (2, num_fields, model.num_values)


def test_relative_2d_attention_forward() -> None:
    d_model, nhead = 32, 4
    attn = Relative2DAttention(d_model, nhead, max_rel_row=3, max_rel_col=3)
    x = torch.randn(1, 16, d_model)
    rows = torch.arange(16) // 4
    cols = torch.arange(16) % 4
    out = attn(x, rows, cols)
    assert out.shape == x.shape

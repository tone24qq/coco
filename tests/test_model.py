import pytest

from model import DynamicMET

torch = pytest.importorskip("torch")


def test_dynamic_met_forward() -> None:
    model = DynamicMET(num_fields=20, num_values=10, rows=4, cols=5)
    x = torch.randint(0, 11, (2, 20))
    out = model(x)
    assert out.shape == (2, 20, 11)

import pytest

from model import DynamicMET

torch = pytest.importorskip("torch")


def test_dynamic_met_forward() -> None:
    num_fields = 20
    model = DynamicMET(num_fields=num_fields, num_values=num_fields, rows=4, cols=5)
    x = torch.randint(0, model.num_values, (2, num_fields))
    out = model(x)
    assert out.shape == (2, num_fields, model.num_values)

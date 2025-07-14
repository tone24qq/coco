import inspect
import sys

from analyzer import predict_scratch_card
from app import GridRequest
from main import parse_args


def test_predict_scratch_card_default_strategy():
    sig = inspect.signature(predict_scratch_card)
    assert sig.parameters["strategy"].default == "outside_in"


def test_grid_request_default_strategy():
    field = GridRequest.model_fields["strategy"]
    assert field.default == "outside_in"


def test_cli_default_strategy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--grid", "1,-1;-1,1"])
    args = parse_args()
    assert args.strategy == "outside_in"

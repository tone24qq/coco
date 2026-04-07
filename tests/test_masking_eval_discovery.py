import numpy as np

from src.masking_eval.discovery import run_module_discovery


class DummyBoard:
    def __init__(self, bid: str, start: int):
        self.board_id = bid
        self.grid = np.arange(start, start + 80).reshape(10, 8)


def test_discovery_handles_insufficient_data() -> None:
    boards = [DummyBoard("b1", 1)]
    result = run_module_discovery(
        boards=boards,
        folds=3,
        repeats=1,
        seed=2026,
        n_trials=1,
        candidate_modules=["local_arith_completion"],
    )
    assert result["insufficient_data"] is True
    assert result["anti_leakage_checks"] == "passed"

from pathlib import Path

from src.masking_eval.data_loader import load_full_boards
from src.masking_eval.discovery import run_module_discovery


def test_module_discovery_runs() -> None:
    boards, _ = load_full_boards(Path("samples/data/full_boards_10x8.json"))
    result = run_module_discovery(
        boards=boards,
        folds=3,
        repeats=1,
        seed=2026,
        n_trials=1,
        candidate_modules=["local_arith_completion"],
    )
    assert result["anti_leakage_checks"] == "passed"
    assert result["num_candidates"] > 0
    assert "champion" in result

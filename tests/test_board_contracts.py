from src.board_contracts import evaluate_board_contract
from src.ticket_specs import build_ticket_spec


def test_contract_duplicate_fail_4x5() -> None:
    spec = build_ticket_spec(4, 5)
    grid = [
        [1, 1, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
    ]
    result = evaluate_board_contract(grid, spec, 0.9, [], strict=True)
    assert result.status == "contract_violation"


def test_contract_legal_range_8x10() -> None:
    spec = build_ticket_spec(8, 10)
    grid = [list(range(r * 10 + 1, r * 10 + 11)) for r in range(8)]
    result = evaluate_board_contract(grid, spec, 0.9, [], strict=True)
    assert result.contract_passed is True


def test_contract_missing_and_illegal() -> None:
    spec = build_ticket_spec(4, 5)
    grid = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 999],
        [16, 17, 18, 19, 20],
    ]
    result = evaluate_board_contract(grid, spec, 0.9, [], strict=True)
    assert result.status == "contract_violation"

from src.ticket_specs import build_ticket_spec, resolve_ticket_spec


def test_build_ticket_spec_4x5_legal_values() -> None:
    spec = build_ticket_spec(4, 5)
    assert min(spec.legal_values) == 1
    assert max(spec.legal_values) == 20


def test_build_ticket_spec_8x10_legal_values() -> None:
    spec = build_ticket_spec(8, 10)
    assert min(spec.legal_values) == 1
    assert max(spec.legal_values) == 80


def test_rows_cols_precedence_over_size_class() -> None:
    spec = resolve_ticket_spec(rows=4, cols=5, size_class="80")
    assert spec.expected_shape == (4, 5)

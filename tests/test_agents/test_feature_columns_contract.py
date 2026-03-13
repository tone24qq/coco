import pytest

from src.utils import V3_CORE20_COLUMNS, validate_feature_columns_contract


def test_v3_contract_passes_with_exact_core20_order() -> None:
    validate_feature_columns_contract(V3_CORE20_COLUMNS, "v3_core20")


def test_v3_contract_fails_when_missing_column() -> None:
    with pytest.raises(ValueError):
        validate_feature_columns_contract(V3_CORE20_COLUMNS[:-1], "v3_core20")


def test_v3_contract_fails_when_wrong_order() -> None:
    cols = V3_CORE20_COLUMNS.copy()
    cols[0], cols[1] = cols[1], cols[0]
    with pytest.raises(ValueError):
        validate_feature_columns_contract(cols, "v3_core20")


def test_v2_contract_not_restricted() -> None:
    validate_feature_columns_contract(["a", "b", "c"], "v2_legacy")

import pytest

from src.normalize_latest import normalize_latest_records


def test_normalize_latest_success() -> None:
    records: list[dict[str, object]] = [
        {
            "issue": "115000001",
            "draw_time": "2026-01-01T00:00:00",
            "numbers": list(range(1, 21)),
        }
    ]
    df = normalize_latest_records(records)
    if len(df) != 1:
        pytest.fail("normalize_latest should return one row")


def test_normalize_latest_schema_mismatch_fail() -> None:
    with pytest.raises(ValueError, match="schema mismatch"):
        normalize_latest_records([{"issue": "1", "numbers": list(range(1, 21))}])

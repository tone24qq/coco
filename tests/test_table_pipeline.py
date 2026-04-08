from src.table_pipeline.main import parse_tables


def test_table_pipeline_parse_tables_smoke() -> None:
    payload = parse_tables("gogo/20/IS23120130.jpg")
    assert payload["is_table_document"] is True
    assert payload["tables"]
    cell = payload["tables"][0]["cells"][0]
    for key in (
        "row_index",
        "col_index",
        "bbox",
        "text",
        "confidence",
        "is_numeric",
        "normalized_value",
        "review_needed",
    ):
        assert key in cell

from pathlib import Path

import src.image_parse_api as image_parse_api
from fastapi.testclient import TestClient

from src.image_parse_api import app


client = TestClient(app)


def test_shape_not_overridden_by_filename_or_image_size(monkeypatch) -> None:
    def _stub_parse_tables(_image_path: str, output_overlay: str | None = None):
        return {
            "tables": [
                {
                    "rows": 10,
                    "cols": 8,
                    "board_bbox": [0, 0, 100, 80],
                    "diagnostics": {"grid_source": "detected_lines", "ocr_backends": ["rapidocr_template_rerank"]},
                    "cells": [
                        {
                            "row_index": r,
                            "col_index": c,
                            "bbox": [c * 10, r * 10, c * 10 + 9, r * 10 + 9],
                            "normalized_value": r * 8 + c + 1,
                            "review_needed": False,
                            "label": "number",
                            "confidence": 0.9,
                            "top_candidates": [{"value": r * 8 + c + 1, "score": 0.9}],
                        }
                        for r in range(10)
                        for c in range(8)
                    ],
                }
            ],
            "overlay_image_base64": None,
            "overlay_path": output_overlay,
        }

    monkeypatch.setattr(image_parse_api, "parse_tables", _stub_parse_tables)

    with Path("gogo/20/IS23120130.jpg").open("rb") as f:
        resp = client.post(
            "/board/parse",
            files={"image": ("NG230516001_頁面_1.jpg", f, "image/jpeg")},
            data={"strict": "true"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"] == 10
    assert data["cols"] == 8

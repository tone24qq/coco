import json
import subprocess
from pathlib import Path

import numpy as np

import scripts.parse_board_image as parse_board_image
from src.grid_detector import GridDetectionError


def test_board_parser_manual_override_schema(tmp_path: Path) -> None:
    manual_grid = tmp_path / "manual.json"
    manual_grid.write_text(
        json.dumps(
            {
                "grid": [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    [17, 18, 19, 20],
                ]
            }
        ),
        encoding="utf-8",
    )
    cmd = [
        "python",
        "scripts/parse_board_image.py",
        "--image",
        "gogo/20/IS23120130.jpg",
        "--size-class",
        "20",
        "--strict",
        "--manual-grid",
        str(manual_grid),
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out.strip().splitlines()[-1])
    for key in (
        "status",
        "source_mode",
        "shape",
        "grid",
        "numbers_all",
        "value_to_position",
        "parse_diagnostics",
        "contract_passed",
        "needs_manual_review",
    ):
        assert key in payload
    assert payload["contract_passed"] is True


def test_board_parser_auto_fail_without_manual() -> None:
    cmd = [
        "python",
        "scripts/parse_board_image.py",
        "--image",
        "gogo/20/IS23120130.jpg",
        "--size-class",
        "20",
        "--strict",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    lines = [x for x in proc.stdout.splitlines() if x.strip().startswith("{")]
    payload = json.loads(lines[-1])
    assert payload["status"] in (
        "needs_manual_review",
        "shape_mismatch",
        "incomplete_grid",
        "low_confidence_parse",
        "contract_violation",
    )


def test_board_parser_override_only_fallback(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "override.json"
    override.write_text(
        json.dumps([{"row": 1, "col": 1, "label": "number", "value": 1}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(parse_board_image.cv2, "imread", lambda *_args, **_kwargs: np.zeros((32, 32), dtype=np.uint8))

    def _raise_detection(*_args, **_kwargs):
        raise GridDetectionError("detect_failed")

    monkeypatch.setattr(parse_board_image, "detect_grid", _raise_detection)
    args = parse_board_image.argparse.Namespace(
        image="dummy.jpg",
        rows=5,
        cols=4,
        size_class=None,
        strict=False,
        manual_grid=None,
        override=str(override),
        query_number=None,
        output_json=str(tmp_path / "out.json"),
        output_csv=str(tmp_path / "out.csv"),
        output_overlay=str(tmp_path / "out.png"),
        no_overlay=True,
    )
    payload = parse_board_image.parse_image_hybrid(args)
    assert payload["source_mode"] == "hybrid"
    assert payload["grid"][0][0] == 1


def test_board_parser_auto_selects_shape(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        parse_board_image.cv2,
        "imread",
        lambda *_args, **_kwargs: np.zeros((64, 64), dtype=np.uint8),
    )

    def _detect(_gray, spec):
        if spec.expected_shape == (5, 4):
            return parse_board_image.GridDetection(
                board_image=np.zeros((320, 256), dtype=np.uint8),
                board_bbox=(0, 0, 10, 10),
                row_count=5,
                col_count=4,
                row_lines=[0, 64, 128, 192, 256, 320],
                col_lines=[0, 64, 128, 192, 256],
                board_confidence=0.8,
                warp_confidence=0.8,
                shape_confidence=0.8,
                cell_boxes=[],
            )
        raise GridDetectionError("shape_mismatch")

    monkeypatch.setattr(parse_board_image, "detect_grid", _detect)
    monkeypatch.setattr(
        parse_board_image,
        "structure_board",
        lambda **_kwargs: parse_board_image._manual_result_from_grid(
            image_path="dummy.jpg",
            spec=parse_board_image.get_ticket_spec("20"),
            grid=[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ],
        ),
    )
    args = parse_board_image.argparse.Namespace(
        image="dummy.jpg",
        rows=None,
        cols=None,
        size_class=None,
        strict=False,
        manual_grid=None,
        override=None,
        query_number=None,
        output_json=str(tmp_path / "out.json"),
        output_csv=str(tmp_path / "out.csv"),
        output_overlay=None,
        no_overlay=True,
    )
    payload = parse_board_image.parse_image_hybrid(args)
    assert payload["shape"] == "5x4"

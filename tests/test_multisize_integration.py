from pathlib import Path

import cv2

from src.board_structurer import structure_board
from src.grid_detector import detect_grid
from src.ticket_specs import get_ticket_spec


CASES = [
    ("20", "gogo/20/IS23120130.jpg", "5x4"),
    ("80", "gogo/80/NL230505019.jpg", "10x8"),
    ("120", "gogo/120/NR23010101_頁面_1.jpg", "12x10"),
]


def test_contract_parse_for_sizes() -> None:
    for size_class, image_path, shape in CASES:
        if not Path(image_path).exists():
            continue
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert gray is not None
        spec = get_ticket_spec(size_class)
        det = detect_grid(gray, spec)
        result = structure_board(sample_id=Path(image_path).stem, image_path=image_path, detection=det, spec=spec)
        assert result.shape == shape

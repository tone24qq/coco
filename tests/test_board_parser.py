from pathlib import Path

import cv2

from src.board_structurer import structure_board
from src.grid_detector import detect_grid


def test_board_parser_runs_on_repo_image() -> None:
    image = Path("gogo/20/IS23120130.jpg")
    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    det = detect_grid(gray)
    result = structure_board(sample_id=image.stem, image_path=str(image), detection=det)
    assert result.row_count >= 2
    assert result.col_count >= 2

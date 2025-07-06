import json
import zipfile

from dataset_utils import load_boards_from_zip


def test_load_boards_from_zip(tmp_path):
    zip_path = tmp_path / "boards.zip"
    data = [[1, 2], [3, 4]]
    boards = [data, data]
    filename = "boards_2x2_50000.json"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(filename, json.dumps(boards))

    result = load_boards_from_zip(str(zip_path), 2, 2)
    assert result == boards

# tests/test_cli.py
import subprocess
import sys


def test_cli_quick_exit(tmp_path):
    board = [[1, 2], [3, -1]]
    grid_str = ";".join(",".join(str(x) for x in row) for row in board)
    cp = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--grid",
            grid_str,
            "--iterations",
            "4",
            "--target",
            "4",
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0
    assert "Prediction" in cp.stderr
    assert "Complete!" in cp.stderr


def test_cli_top_k(tmp_path):
    board = [[1, 2], [3, -1]]
    grid_str = ";".join(",".join(str(x) for x in row) for row in board)
    cp = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--grid",
            grid_str,
            "--iterations",
            "4",
            "--target",
            "4",
            "--top-k",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0

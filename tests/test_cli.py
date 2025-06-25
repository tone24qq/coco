# tests/test_cli.py
import subprocess, sys, json, os, textwrap

def test_cli_quick_exit(tmp_path):
    board = [[1, 2], [3, -1]]
    cp = subprocess.run(
        [sys.executable, "main.py", "--grid", json.dumps(board), "--iters", "4"],
        capture_output=True, text=True,
    )
    assert cp.returncode == 0
    assert "Predictions" in cp.stdout
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_check(path: Path) -> int:
    return subprocess.run(
        [sys.executable, "check_npz.py", str(path)], capture_output=True
    ).returncode


def test_check_npz_cli_ok(tmp_path):
    f = tmp_path / "2x2.npz"
    freq = np.zeros((2, 2, 5), dtype=np.uint16)
    meta = {"schema_version": 1, "generated_at": "now"}
    np.savez(f, freq=freq, meta=meta)
    assert run_check(f) == 0


def test_check_npz_cli_fail(tmp_path):
    f = tmp_path / "bad.npz"
    np.savez(f, foo=np.zeros(1))
    assert run_check(f) == 1

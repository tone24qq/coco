import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session", autouse=True)
def _build_npz() -> None:
    """Ensure memory caches are generated before running tests."""
    npz_path = Path("data_archives/4x5_memory.npz")
    if npz_path.exists():
        return
    env = os.environ.copy()
    env.setdefault("MEMORY_SAMPLE_LIMIT", "5")
    subprocess.run([sys.executable, "build_memories.py"], check=True, env=env)

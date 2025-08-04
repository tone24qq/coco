import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_memories  # noqa: E402

# Ensure memory caches exist for tests (only small fixtures)
build_memories.build_all_memories(shapes={"4x5"})

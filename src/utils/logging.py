"""Simple JSON logger."""

from __future__ import annotations

import json
from typing import Any


def log(event: str, **kwargs: Any) -> None:
    record = {"event": event, **kwargs}
    print(json.dumps(record))

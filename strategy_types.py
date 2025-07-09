from enum import Enum


class Strategy(Enum):
    """Available prediction strategies."""

    LEGACY = "legacy"
    MODERN = "modern"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

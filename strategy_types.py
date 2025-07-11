from enum import Enum


class Strategy(Enum):
    """Available prediction strategies."""

    LEGACY = "legacy"
    MODERN = "modern"
    SAMPLE_LINE = "sample_line"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

"""Common utilities for CSP operations."""

from __future__ import annotations


def get_subgrid_indices(row: int, col: int, sub: int) -> tuple[int, int]:
    """Return the top-left corner (r0, c0) of subgrid containing ``(row, col)``."""
    return ((row // sub) * sub, (col // sub) * sub)

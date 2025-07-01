# isort: skip_file
# tests/test_registry_references.py
# This test ensures static analyzers (e.g., Vulture) and coverage tools
# see all registered strategy functions as used, preventing false dead-code flags.

import pytest

from modules import (
    compute_difference_trend,
    compute_focus_score,
    connectivity_heatmap,
    detect_mirror_sequences,
    detect_skip_patterns,
    entropy_spread_score,
    gradient_affinity,
    row_col_bias,
    sequence_tail_analyzer,
    target_affinity,
)


@pytest.mark.parametrize(
    "fn",
    [
        compute_focus_score,
        detect_skip_patterns,
        compute_difference_trend,
        detect_mirror_sequences,
        connectivity_heatmap,
        sequence_tail_analyzer,
        target_affinity,
        gradient_affinity,
        row_col_bias,
        entropy_spread_score,
    ],
)
def test_function_is_callable(fn):
    # Reference each function by name, so static analysis sees usage
    assert callable(fn)

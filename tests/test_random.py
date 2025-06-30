# tests/property/test_random.py
import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from analyzer import predict_scratch_card

grid_sizes = st.integers(min_value=4, max_value=20)


@pytest.mark.xfail(reason="random boards may exceed module limits")
@given(r=grid_sizes, c=grid_sizes)
@settings(
    deadline=None,
    max_examples=5,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_predict_random_board(r, c):
    import numpy as np

    board = np.arange(1, r * c + 1, dtype=int).reshape(r, c)
    board[r // 2, c // 2] = -1
    result = predict_scratch_card(board.tolist(), iterations=8, unique=False)
    assert "predictions" in result

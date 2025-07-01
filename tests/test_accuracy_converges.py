import pytest

from tests.reliability_utils import run_until_converged


@pytest.mark.timeout(120)
def test_accuracy_estimation_converges():
    p, hw, total = run_until_converged(
        min_size=4,
        max_size=6,
        batch_size=50,
        delta=0.1,
        seed=123,
    )
    assert total > 0
    assert 0.0 <= p <= 1.0
    assert hw <= 0.1

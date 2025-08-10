import random

import numpy as np
import torch

from src.utils.seed import seed_all


def test_seed_all_produces_deterministic_results():
    seed_all(123)
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    seed_all(123)
    assert r1 == random.random()
    assert n1 == np.random.rand()
    assert t1 == torch.rand(1).item()
    assert torch.are_deterministic_algorithms_enabled()

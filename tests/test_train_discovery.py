import random

import numpy as np
import torch

from train import set_seed


def test_set_seed_reproducible():
    set_seed(123)
    r1 = torch.rand(2)
    n1 = np.random.rand(2)
    py1 = random.random()

    set_seed(123)
    r2 = torch.rand(2)
    n2 = np.random.rand(2)
    py2 = random.random()

    assert torch.allclose(r1, r2)
    assert np.allclose(n1, n2)
    assert py1 == py2

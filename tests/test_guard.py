import numpy as np

from utils.guard import index_to_coord


def test_index_to_coord_matches_numpy() -> None:
    shape = (4, 5)
    for idx in range(shape[0] * shape[1]):
        assert index_to_coord(idx, shape) == tuple(np.unravel_index(idx, shape))

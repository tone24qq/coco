import numpy as np

import analyzer


def test_dtype_for_shape():
    assert analyzer.dtype_for_shape(5, 5) == np.uint16
    assert analyzer.dtype_for_shape(20, 20) == np.uint32
    assert analyzer.dtype_for_shape(40, 10) == np.int64
    assert analyzer.dtype_for_shape(31, 31) == np.int64

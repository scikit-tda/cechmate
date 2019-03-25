import numpy as np

from cechmate.utils import sparse_to_dense


def test_sparse_bm_to_dense():
    sparse = [
        (0, []),
        (0, []),
        (1, [0, 1]),
        (0, []),
        (1, [1, 3])
    ]    
    expected = np.array([
        [0,0,1,0,0],
        [0,0,1,0,1],
        [0,0,0,0,0],
        [0,0,0,0,1],
        [0,0,0,0,0]
    ], np.float32)

    dense = sparse_to_dense(sparse)
    np.testing.assert_array_equal(dense, expected)

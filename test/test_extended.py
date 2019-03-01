import numpy as np
import pytest

from cechmate import extended


@pytest.fixture
def X():
    f = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
    }

    X = [
        [0], [1], [2], [3], [4], [5], [6],
        [0,2], [1,3], [2,4], [3,4], [4,5], [4,6]
    ]

    return X, f

@pytest.fixture
def triangle():
    X = [
        [0],[1],[2],
        [0,1],[1,2],[0,2],
        [0,1,2]
    ]
    f = {0:4.1, 1:1.1, 2:0.1}

    return X, f

def test_lower_star(triangle):
    X, f = triangle
    lstar = extended.lower_star(X, 0, f)

    assert len(lstar) == 4
    assert [0,1,2] in lstar

    f = {0:4, 1:1, 2:1}
    lstar = extended.lower_star(X, 1, f)

    assert len(lstar) == 2
    assert [0,1,2] not in lstar

def test_lower_boundary_matrix(triangle):
    X, f = triangle

    bm, _ = extended.up_down_boundary_matrix(X, f)
    assert bm == [
        (0, []), 
        (0, []), 
        (1, [0, 1]), 
        (0, []), 
        (1, [1, 3]), 
        (1, [0, 3]), 
        (1, [3]), 
        (1, [1]), 
        (2, [4, 6, 7]), 
        (1, [0]), 
        (2, [2, 7, 9]), 
        (2, [5, 6, 9])
    ]

def test_reduction(triangle):
    X, f = triangle
    bm, _ = extended.up_down_boundary_matrix(X, f)

    red_bm = extended.reduce_boundary_matrix(bm)
    assert red_bm == [
        (0, []),
        (0, []),
        (1, [0, 1]),
        (0, []),
        (1, [1, 3]),
        (1, []),
        (2, [2, 4, 5]),
        (1, [0]),
        (1, []),
        (2, [4, 7, 8]),
        (1, []),
        (2, [2, 8, 10]),
        (2, []),
        (3, [6, 9, 11, 12])
    ]

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

    dense = extended.sparse_bm_to_dense(sparse)
    np.testing.assert_array_equal(dense, expected)

def test_bm_separation(triangle):
    red_bm = [
        (0, []),
        (0, []),
        (1, [0, 1]),
        (0, []),
        (1, [1, 3]),
        (1, []),
        (2, [2, 4, 5]),
        (1, [0]),
        (1, []),
        (2, [4, 7, 8]),
        (1, []),
        (2, [2, 8, 10]),
        (2, []),
        (3, [6, 9, 11, 12])
    ]

    ord_expect = [
        (0, []),
        (0, []),
        (1, [0, 1]),
        (0, []),
        (1, [1, 3]),
        (1, []),
        (2, [2, 4, 5]) 
    ]

    ext_expect = [
        (1, [0]),
        (1, []),
        (2, [4]),
        (1, []),
        (2, [2]),
        (2, []),
        (3, [6])
    ]

    rel_expect = [
        (1, []),
        (1, []),
        (2, [0, 1]),
        (1, []),
        (2, [1, 3]),
        (2, []),
        (3, [2, 4, 5])
    ]

    # import pdb; pdb.set_trace()
    ordinary, ext, relative = extended.separate_boundary_matrix(red_bm)

    assert ordinary == ord_expect
    assert ext == ext_expect

    import pdb; pdb.set_trace()
    assert relative == rel_expect





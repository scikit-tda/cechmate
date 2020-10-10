import numpy as np

__all__ = ["sparse_to_dense"]


def sparse_to_dense(sparse_bm):
    """Converts a sparse boundary matrix to a dense boundary matrix.

    This is used for visualization and debugging as dense boundary matrices can be easier to read to small matrices. Dense boundary matrices are the default filtration format used in cechmate filtrations.

    Parameters
    ============

    sparse_bm: 
        Sparse boundary matrix.

    Returns
    =========

    dense: np.array
        Square matrix representation of boundary matrix.

    """
    n = len(sparse_bm)
    dense = np.zeros((n, n))

    for i, (_, c) in enumerate(sparse_bm):
        dense[c, i] = 1

    return dense

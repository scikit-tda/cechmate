"""Goal: Develop a framework for Extended Persistence.

We can use Phat boundary matrix reduction for the actual reduction, but will need to build
    - convert an abstract simplicial complex to the correct boundary matrix
    - read the reduced boundary matrix into birth-death pairs.

"""

"""Given: a simplicial complex with birth-times encoded as the level set

- Compute the upward pass
- Compute the downward pass

"""



"""

    5       6 
      \   /  
        4
      /   #
    2       3
    |       |
    0       1

"""

import numpy as np
import phat

from .solver import _simplices_to_sparse_pivot_column, _process_distances, _add_unpaired





def star(X, v):
    """Compute star of v
    """
    st = [x for x in X if v in x]
    return st

def lower_star(X, v, f):
    st = star(X, v)
    lst = [x for x in st if max([f[y] for y in x]) == f[v]]
    return lst

def upper_star(X, v, f):
    st = star(X, v)
    lst = [x for x in st if min([f[y] for y in x]) == f[v]]
    return lst


def up_down_boundary_matrix(X, f):
    """
        Let A be the boundary matrix for the ascending pass, storing the simplices in blocks that correspond to the lower stars of v1 to vn, in this order.

        All simplices in the same block are assigned the same value, namely the height of the vertex defining the lower star.
    """
    

    vs = [x[0] for x in X if len(x) == 1]
    fvs = sorted(vs, key=lambda v: f[v])
    lstars = [(lower_star(X, v, f), f[v]) for v in fvs]
    kappas = [(kappa, fv) for lstar, fv in lstars for kappa in sorted(lstar, key=len)]

    ustars = [(upper_star(X, v, f), f[v]) for v in fvs[::-1]]
    lambdas = [(lam, fv) for ustar, fv in ustars for lam in sorted(ustar, key=len)]

    A = _simplices_to_sparse_pivot_column(kappas)
    D = _simplices_to_sparse_pivot_column(lambdas)

    # Augment D by lowering it it by m and coning.
    M = list(A)

    kap_sims = [l[0] for l in kappas]
    for (k, ds), lam in zip(D, lambdas):
        # find index of lam in A (or kappas)
        idx = kap_sims.index(lam[0])
        M.append(((k+1), [idx] + [len(A) + d for d in ds]))
    
    return M, kappas + lambdas

def reduce_boundary_matrix(M):
    boundary_matrix = phat.boundary_matrix(
        columns = M, 
        representation = phat.representations.sparse_pivot_column
    )


    # bm = phat.reductions.chunk_reduction(boundary_matrix)
    bm = boundary_matrix.compute_reduction()
    # pairs.sort()
    bm = [(c.dimension, c.boundary) for c in bm.columns]
    
    return bm

def sparse_bm_to_dense(sparse_bm):
    """Return the sparse boundary matrix as a dense boundary matrix.

    This is used for visualization and debugging as dense boundary matrices are subjectively easier to read.
    """
    n = len(sparse_bm)
    dense = np.zeros((n,n))
    
    for i, (_, c) in enumerate(sparse_bm):
        dense[c, i] = 1 

    return dense


def separate_boundary_matrix(bm):
    """
    Note from CT:
    - For A, the birth values increase downward and the death values from left to right, so we need to turn the quadrant by 90◦ to get the ordinary sub-diagram. 
    
    - Symmetrically, we turn the quadrant of B by −90◦ to get the relative sub- diagram and 
    - we reflect the quadrant of P across the main diagonal to get the extended sub-diagram. 
    
    Since the reduced versions of A and B are upper triangular, we indeed get the ordinary sub-diagram above and the relative sub- diagram below the diagonal.
    """

    n = int(len(bm) / 2)
    assert len(bm) / 2 == n, "bm should have even dimension for both up and down pass."
    ordinary = bm[:n]
    extended = [(k, [c for c in cs if c < n]) for k, cs in bm[n:]]
    relative = [(k, [c-n for c in cs if c >= n]) for k, cs in bm[n:]]


    return ordinary, extended, relative


    # for b in bm:
        # print(b)



    # print(pairs)
    ## Setup persistence diagrams by reading off distances
    # dgms = _process_distances(pairs, simplices)
    # print(dgms)

    # dgms = _add_unpaired(dgms, pairs, simplices)
 

    # Then call `_simplices_to_sparse_pivot_column`for kappas and lambdas
    #  - the result will be A and B
    #  - then construct P and call it good.


    



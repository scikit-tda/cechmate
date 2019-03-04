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



def diagrams(X, f):
    bm, mapping = up_down_boundary_matrix(X, f)
    mapping = dict(enumerate(mapping))
    pairs = reduce_boundary_matrix(bm)

    n = len(bm) / 2
    ordinary_pairs = [(b,d) for (b,d) in pairs if b < n and d < n]
    extended_pairs = [(b,d) for (b,d) in pairs if b < n and d >= n]
    relative_pairs = [(b,d) for (b,d) in pairs if b >= n and d >= n]

    # print(f"Ordinary: {ordinary_pairs}")
    # print(f"Extended: {extended_pairs}")
    # print(f"Relative: {relative_pairs}")

    # for ps in [ordinary_pairs, extended_pairs, relative_pairs]:
    #     res = [(mapping[b], mapping[d]) for b,d in ps]
    #     # print(f"Mapping results:")
    #     for x in res: print(f"\t{x}")

    # assert len(ordinary_pairs) + len(extended_pairs) + len(relative_pairs) == len(pairs)

    diagrams = {}

    for b,d in ordinary_pairs:
        order = len(mapping[b][0]) - 1
        s = "ordinary"
        diagrams.setdefault(order , {}).setdefault(s, []).append((mapping[b][1], mapping[d][1]))

    for b,d in extended_pairs:
        order = len(mapping[b][0]) - 1
        s = "extended"
        diagrams.setdefault(order , {}).setdefault(s, []).append((mapping[b][1], mapping[d][1]))
        
    for b,d in relative_pairs:
        order = len(mapping[d][0]) - 1 # order computation is different!
        s = "relative"
        diagrams.setdefault(order , {}).setdefault(s, []).append((mapping[b][1], mapping[d][1]))
    
    diagrams = {h: {s:[(b,d) for b,d in ls if b != d] for s, ls in d.items()} for h,d in diagrams.items()}



    return diagrams


def example():
    """ This example taken from the reeb graph in Carri`ere 2017 of Figure 4"""

    f = {
        1: 0.0,
        2: 0.5,
        3: 1.0,
        4: 1.5,
        5: 1.5,
        6: 2.0,
        7: 2.0,
        8: 2.5, 
        9: 3.0,
        10: 3.5
    }

    X = [
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
        [1,3], [2,4], [3,4], [3,5], [4,6], [5,7], [7,9], [7,8], [6,8], [8,10]
    ]

    expected = {
        0: {
            "ordinary": [[0.5, 1.5]],
            "extended": [[0.0, 3.5]],
            "relative": [[]]
        },
        1: {
            "ordinary": [[]],
            "extended": [[2.5, 1.0]],
            "relative": [[3.0, 2.0]]
        }
    }

    return X, f, expected



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

        Returns
        --------

        boundary matrix: sparse pivot column boundary matrix
        f: mapping of simplices to function values
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

    pairs = boundary_matrix.compute_persistence_pairs()
    pairs.sort()

    # pairs = [for bi, di in pairs]
    return list(pairs)

    # # bm = phat.reductions.chunk_reduction(boundary_matrix)
    # bm = boundary_matrix.compute_reduction()

    # bm = [(c.dimension, c.boundary) for c in bm.columns]
    
    # return bm

def sparse_bm_to_dense(sparse_bm):
    """Return the sparse boundary matrix as a dense boundary matrix.

    This is used for visualization and debugging as dense boundary matrices are subjectively easier to read.
    """
    n = len(sparse_bm)
    dense = np.zeros((n,n))
    
    for i, (_, c) in enumerate(sparse_bm):
        dense[c, i] = 1 

    return dense


def get_diagrams(bm, f):
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

    # import pdb; pdb.set_trace()
    relative = [(k, [c for c in cs if c >= n]) for k, cs in bm[n:]]


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


    



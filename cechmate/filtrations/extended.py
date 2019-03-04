


import numpy as np
import phat

from .base import BaseFiltration

from ..solver import _simplices_to_sparse_pivot_column

__all__ = ["Extended"]

class Extended(BaseFiltration):
    """
    This class computed the extended persistence of a simplicial complex. It requires input as a simplicial complex and a mapping on each vertex in the complex. It returns a dictionary storing the associated diagrams in each homology class.

    The basic steps are:
        - convert an abstract simplicial complex to the correct boundary matrix, using the lower-star up pass and upper-star down pass
        - read the reduced boundary matrix into birth-death pairs.
        - partition pairs into respective Ordinary/Extended/Relative diagrams.
    """

    def __init__(self):
        self._boundary_matrix = None
        self._mapping = None
        self._reduced_boundary_matrix = None
        self._pairs = None
        self.diagrams_ = None

    def diagrams(self, X, f):
        """ Compute diagrams of extended persistent homology.
        """
        _, _ = self._up_down_boundary_matrix(X, f)
        pairs = self._compute_persistence_pairs()
        diagrams = self._process_pairs(pairs)

        self.diagrams_ = diagrams
        return self.diagrams_

    def _process_pairs(self, pairs):

        n = len(self._boundary_matrix) / 2
        ordinary_pairs = [(b,d) for (b,d) in pairs if b < n and d < n]
        extended_pairs = [(b,d) for (b,d) in pairs if b < n and d >= n]
        relative_pairs = [(b,d) for (b,d) in pairs if b >= n and d >= n]

        diagrams = {}

        self._extract_diagram(
            diagrams, 
            ordinary_pairs, 
            "ordinary", 
            lambda b,d: len(self._mapping[b][0]) - 1
        )
        self._extract_diagram(
            diagrams, 
            extended_pairs, 
            "extended", 
            lambda b,d: len(self._mapping[b][0]) - 1
        )

        self._extract_diagram(
            diagrams, 
            relative_pairs, 
            "relative", 
            lambda b,d: len(self._mapping[d][0]) - 1
        )

        diagrams = {
            h: {
                s:[[b,d] for b,d in ls if b != d] for s, ls in d.items()
            } for h,d in diagrams.items()
        }

        return diagrams

    def _extract_diagram(self, diagrams, pairs, pairs_str, order_f):
        for b,d in pairs:
            order = order_f(b,d)
            diagrams.setdefault(order , {}).setdefault(pairs_str, []).append((self._mapping[b][1], self._mapping[d][1]))

    def _up_down_boundary_matrix(self, X, f):
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

        lstars = [(_lower_star(X, v, f), f[v]) for v in fvs]
        kappas = [(kappa, fv) for lstar, fv in lstars for kappa in sorted(lstar, key=len)]

        ustars = [(_upper_star(X, v, f), f[v]) for v in fvs[::-1]]
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
        
        self._boundary_matrix = M
        self._mapping = dict(enumerate(kappas + lambdas))
        return self._boundary_matrix, self._mapping 

    def _compute_persistence_pairs(self, boundary_matrix=None):
        boundary_matrix = boundary_matrix or self._boundary_matrix
        
        self._reduced_boundary_matrix = phat.boundary_matrix(
            columns = boundary_matrix, 
            representation = phat.representations.sparse_pivot_column
        )

        pairs = self._reduced_boundary_matrix.compute_persistence_pairs()
        pairs.sort()
        self._pairs = list(pairs)
        return self._pairs


    @staticmethod
    def sparse_bm_to_dense(sparse_bm):
        """Return the sparse boundary matrix as a dense boundary matrix.

        This is used for visualization and debugging as dense boundary matrices are subjectively easier to read.
        """
        n = len(sparse_bm)
        dense = np.zeros((n,n))
        
        for i, (_, c) in enumerate(sparse_bm):
            dense[c, i] = 1 

        return dense


def _star(X, v):
    """Compute star of v
    """
    st = [x for x in X if v in x]
    return st

def _lower_star(X, v, f):
    st = _star(X, v)
    lst = [x for x in st if max([f[y] for y in x]) == f[v]]
    return lst

def _upper_star(X, v, f):
    st = _star(X, v)
    lst = [x for x in st if min([f[y] for y in x]) == f[v]]
    return lst

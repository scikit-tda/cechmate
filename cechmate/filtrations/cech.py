import itertools
import numpy as np

from .base import BaseFiltration
from .miniball import miniball_cache


__all__ = ["Cech"]


class Cech(BaseFiltration):
    """Compute the Cech filtration of a Euclidean point set for simplices up to order :code:`self.max_dim`.

    Examples
    ========

        >>> r = Cech(maxdim=1)
        >>> simplices = r.build(X)
        >>> diagrams = r.diagrams(simplices)

    """

    def build(self, X):
        """Compute the Cech filtration of a Euclidean point set for simplices up to order :code:`self.max_dim`.

        Parameters
        ===========
        
        X: Nxd array
            N Euclidean vectors in d dimensions

        Returns
        ==========
        
        simplices: 
            Cech filtration for the data X
        """

        N = X.shape[0]
        xr = np.arange(N)
        xrl = xr.tolist()
        maxdim = self.maxdim
        if not self.maxdim:
            maxdim = X.shape[1] - 1

        miniball = miniball_cache(X)

        # start with vertices
        simplices = [([i], 0) for i in range(N)]

        # then higher order simplices
        for k in range(maxdim + 1):
            for idxs in itertools.combinations(xrl, k + 2):
                C, r2 = miniball(frozenset(idxs), frozenset([]))
                simplices.append((list(idxs), np.sqrt(r2)))

        self.simplices_ = simplices

        return simplices

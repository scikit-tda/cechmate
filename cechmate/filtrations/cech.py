import itertools
import numpy as np

from .base import BaseFiltration
from .miniball import miniball_cache

class Cech(BaseFiltration):
    def build(self, X):
        """
        Compute the Cech filtration of a Euclidean point set for simplices up to order `self.max_dim`.

        :param X: An Nxd array of N Euclidean vectors in d dimensions

        :returns simplices: Cech filtration for the data X
        """

        N = X.shape[0]
        xr = np.arange(N)
        xrl = xr.tolist()

        miniball = miniball_cache(X)

        # start with vertices
        simplices = [([i], 0) for i in range(N)]

        # then higher order simplices
        for k in range(self.max_dim + 1):
            for idxs in itertools.combinations(xrl, k + 2):
                C, r2 = miniball(frozenset(idxs), frozenset([]))
                simplices.append((list(idxs), np.sqrt(r2)))

        return simplices






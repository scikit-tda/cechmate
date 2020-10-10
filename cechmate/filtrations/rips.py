import itertools
import numpy as np

from .base import BaseFiltration

__all__ = ["Rips"]


class Rips(BaseFiltration):
    """Construct a Rips filtration and the associated diagrams.

    Examples
    ========

        >>> r = Rips(maxdim=1)
        >>> simplices = r.build(X)
        >>> diagrams = r.diagrams(simplices)

    """

    def build(self, X):
        """Compute the rips filtration of a Euclidean point set.

        Parameters
        ===========
        X: An Nxd array
            An Nxd array of N Euclidean vectors in d dimensions.

        Returns
        ========
        simplices: list of tuples
            List of simplices with birth time representing Rips filtration for the data X.
        """
        D = self._getSSM(X)
        N = D.shape[0]
        xr = np.arange(N)
        xrl = xr.tolist()
        maxdim = self.maxdim
        if not maxdim:
            maxdim = 1
        # First add all 0 simplices
        simplices = [([i], 0) for i in range(N)]
        for k in range(maxdim + 1):
            # Add all (k+1)-simplices, which have (k+2) vertices
            for idxs in itertools.combinations(xrl, k + 2):
                idxs = list(idxs)
                d = 0.0
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        d = max(d, D[idxs[i], idxs[j]])
                simplices.append((idxs, d))

        self.simplices_ = simplices

        return simplices

    def _getSSM(self, X):
        """
        Given a set of Euclidean vectors, return a pairwise distance matrix
        :param X: An Nxd array of N Euclidean vectors in d dimensions
        :returns D: An NxN array of all pairwise distances
        """
        XSqr = np.sum(X ** 2, 1)
        D = XSqr[:, None] + XSqr[None, :] - 2 * X.dot(X.T)
        D[D < 0] = 0  # Numerical precision
        D = np.sqrt(D)
        return D

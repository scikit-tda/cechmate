import itertools
import numpy as np

from .base import BaseFiltration


class Rips(BaseFiltration):
    def build(self, X):
        """
        Do the rips filtration of a Euclidean point set

        :param X: An Nxd array of N Euclidean vectors in d dimensions

        :returns simplices: Rips filtration for the data X
        """
        D = self.getSSM(X)
        N = D.shape[0]
        xr = np.arange(N)
        xrl = xr.tolist()
        # First add all 0 simplices
        simplices = [([i], 0) for i in range(N)]
        for k in range(self.max_dim + 1):
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

    def getSSM(self, X):
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


__all__ = ["Rips"]

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
            maxdim = X.shape[1]

        miniball = miniball_cache(X)

        # start with vertices
        simplices = [([i], 0) for i in range(N)]

        # then higher order simplices
        for k in range(maxdim+1):
            for idxs in itertools.combinations(xrl, k + 1):
                C, r2 = miniball(frozenset(idxs), frozenset([]))
                simplices.append((list(idxs), np.sqrt(r2)))

        self.simplices_ = simplices

        return simplices

    def build_thresh(self, X, r_max=20):
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
            maxdim = X.shape[1]

        miniball = miniball_cache(X)

        # start with vertices
        simplices = [([i], 0) for i in range(N)]

        #then insert edges (proximity graph)
        prox_graph=[]
        for i in range(N):
            for j in range(i):
                d=np.linalg.norm(X[i]-X[j])
                if d/2<r_max:
                    simplices.append(([j,i], d/2))
                    prox_graph.append([j,i])




        for k in range(2, maxdim+1): # then higher order simplices
            print("test_1")

            #For each k-clique, check if the edges are in the proximity graph
            for idxs in itertools.combinations(xrl, k+1):
                candidate=True
                j=0
                while candidate and j!= k-1:
                    for i in range(k+1):
                        for j in range(i):
                            if [idxs[j], idxs[i]] not in prox_graph:
                                candidate = False
                if candidate:
                    C, r2=miniball(frozenset(idxs), frozenset([]))
                    if np.sqrt(r2) < r_max:
                        simplices.append((list(idxs), np.sqrt(r2)))


        self.simplices_ = simplices
        return simplices




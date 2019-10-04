import itertools
import time

import numpy as np
import numpy.linalg as linalg
from scipy import spatial

from .base import BaseFiltration
from .miniball import miniball_cache


__all__ = ["Alpha"]


class Alpha(BaseFiltration):
    """ Construct an Alpha filtration from the given data.

    Note
    =====

    Alpha filtrations use radius instead of diameter. Multiply results or X by 2 when comparing the filtration to Rips or Cech.

    Examples
    ========

        >>> r = Alpha()
        >>> simplices = r.build(X)
        >>> diagrams = r.diagrams(simplices)

    """

    def build(self, X):
        """
        Do the Alpha filtration of a Euclidean point set (requires scipy)
        
        Parameters
        ===========
        X: Nxd array
            Array of N Euclidean vectors in d dimensions
        """

        if X.shape[0] < X.shape[1]:
            warnings.warn(
                "The input point cloud has more columns than rows; "
                + "did you mean to transpose?"
            )
        maxdim = self.maxdim
        if self.maxdim is None:
            maxdim = X.shape[1]

        ## Step 1: Figure out the filtration
        if self.verbose:
            print("Doing spatial.Delaunay triangulation...")
            tic = time.time()

        delaunay_faces = spatial.Delaunay(X).simplices
        
        if self.verbose:
            print(
                "Finished spatial.Delaunay triangulation (Elapsed Time %.3g)"
                % (time.time() - tic)
            )
            print("Building alpha filtration...")
            tic = time.time()
	
        miniball = miniball_cache(X)
        filtration = {}
        for dim in range(maxdim, 0, -1):
            for s in range(delaunay_faces.shape[0]):
                simplex = delaunay_faces[s, :]
                for sigma in itertools.combinations(simplex, dim + 1):
                    sigma = tuple(sorted(sigma))
                    if not sigma in filtration:
                        C, r2 = miniball(frozenset(sigma), frozenset([]))
                        filtration[sigma] = np.sqrt(r2)

        ## Step 2: Take care of numerical artifacts that may result
        ## in simplices with greater filtration values than their co-faces
        for dim in range(maxdim, 1, -1):
            for sigma in filter(lambda x: len(x) == dim + 1, filtration.keys()):
                for i in range(len(sigma)):
                    tau = sigma[:i] + sigma[i + 1:]
                    if filtration[tau] > filtration[sigma]:
                        filtration[tau] = filtration[sigma]
        
        if self.verbose:
            print(
                "Finished building alpha filtration (Elapsed Time %.3g)"
                % (time.time() - tic)
            )

        simplices = [([i], 0) for i in range(X.shape[0])]
        simplices.extend([(list(simplex), filtr) for (simplex, filtr) in filtration.items()])

        self.simplices_ = simplices

        return simplices


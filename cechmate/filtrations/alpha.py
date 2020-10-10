import itertools
import time
import warnings

import numpy as np
from numpy import linalg
from scipy import spatial

from .base import BaseFiltration

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

    MIN_DET = 1e-10

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
        if not self.maxdim:
            maxdim = X.shape[1] - 1

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

        filtration = {}
        for dim in range(maxdim + 2, 1, -1):
            for s in range(delaunay_faces.shape[0]):
                simplex = delaunay_faces[s, :]
                for sigma in itertools.combinations(simplex, dim):
                    sigma = tuple(sorted(sigma))
                    if not sigma in filtration:
                        rSqr = self._get_circumcenter(X[sigma, :])[1]
                        if np.isfinite(rSqr):
                            filtration[sigma] = rSqr
                    if sigma in filtration:
                        for i in range(dim):  # Propagate alpha filtration value
                            tau = sigma[0:i] + sigma[i + 1 : :]
                            if tau in filtration:
                                filtration[tau] = min(
                                    filtration[tau], filtration[sigma]
                                )
                            elif len(tau) > 1 and sigma in filtration:
                                # If Tau is not empty
                                xtau, rtauSqr = self._get_circumcenter(X[tau, :])
                                if np.sum((X[sigma[i], :] - xtau) ** 2) < rtauSqr:
                                    filtration[tau] = filtration[sigma]
        # Convert from squared radii to radii
        for sigma in filtration:
            filtration[sigma] = np.sqrt(filtration[sigma])

        ## Step 2: Take care of numerical artifacts that may result
        ## in simplices with greater filtration values than their co-faces
        simplices_bydim = [set([]) for i in range(maxdim + 2)]
        for simplex in filtration.keys():
            simplices_bydim[len(simplex) - 1].add(simplex)
        simplices_bydim = simplices_bydim[2::]
        simplices_bydim.reverse()
        for simplices_dim in simplices_bydim:
            for sigma in simplices_dim:
                for i in range(len(sigma)):
                    tau = sigma[0:i] + sigma[i + 1 : :]
                    if filtration[tau] > filtration[sigma]:
                        filtration[tau] = filtration[sigma]

        if self.verbose:
            print(
                "Finished building alpha filtration (Elapsed Time %.3g)"
                % (time.time() - tic)
            )

        simplices = [([i], 0) for i in range(X.shape[0])]
        simplices.extend(filtration.items())

        self.simplices_ = simplices

        return simplices

    def _get_circumcenter(self, X):
        """
        Compute the circumcenter and circumradius of a simplex
        
        Parameters
        ----------
        X : ndarray (N, d)
            Coordinates of points on an N-simplex in d dimensions
        
        Returns
        -------
        (circumcenter, circumradius)
            A tuple of the circumcenter and squared circumradius.  
            (SC1) If there are fewer points than the ambient dimension plus one,
            then return the circumcenter corresponding to the smallest
            possible squared circumradius
            (SC2) If the points are not in general position, 
            it returns (np.inf, np.inf)
            (SC3) If there are more points than the ambient dimension plus one
            it returns (np.nan, np.nan)
        """
        X0 = np.array(X)
        if X.shape[0] == 2:
            # Special case of an edge, which is very simple
            dX = X[1, :] - X[0, :]
            rSqr = 0.25 * np.sum(dX ** 2)
            x = X[0, :] + 0.5 * dX
            return (x, rSqr)
        if X.shape[0] > X.shape[1] + 1:  # SC3 (too many points)
            warnings.warn(
                "Trying to compute circumsphere for "
                + "%i points in %i dimensions" % (X.shape[0], X.shape[1])
            )
            return (np.nan, np.nan)
        # Transform arrays for PCA for SC1 (points in higher ambient dimension)
        muV = np.array([])
        V = np.array([])
        if X.shape[0] < X.shape[1] + 1:  # SC1: Do PCA down to NPoints-1
            muV = np.mean(X, 0)
            XCenter = X - muV
            _, V = linalg.eigh((XCenter.T).dot(XCenter))
            V = V[:, (X.shape[1] - X.shape[0] + 1) : :]  # Put dimension NPoints-1
            X = XCenter.dot(V)
        muX = np.mean(X, 0)
        D = np.ones((X.shape[0], X.shape[0] + 1))
        # Subtract off centroid and scale down for numerical stability
        Y = X - muX
        scaleSqr = np.max(np.sum(Y ** 2, 1))
        scaleSqr = 1
        scale = np.sqrt(scaleSqr)
        Y /= scale

        D[:, 1:-1] = Y
        D[:, 0] = np.sum(D[:, 1:-1] ** 2, 1)
        minor = lambda A, j: A[
            :, np.concatenate((np.arange(j), np.arange(j + 1, A.shape[1])))
        ]
        dxs = np.array([linalg.det(minor(D, i)) for i in range(1, D.shape[1] - 1)])
        alpha = linalg.det(minor(D, 0))
        if np.abs(alpha) > Alpha.MIN_DET:
            signs = (-1) ** np.arange(len(dxs))
            x = dxs * signs / (2 * alpha) + muX  # Add back centroid
            gamma = ((-1) ** len(dxs)) * linalg.det(minor(D, D.shape[1] - 1))
            rSqr = (np.sum(dxs ** 2) + 4 * alpha * gamma) / (4 * alpha * alpha)
            x *= scale
            rSqr *= scaleSqr
            if V.size > 0:
                # Transform back to ambient if SC1
                x = x.dot(V.T) + muV
            return (x, rSqr)
        return (np.inf, np.inf)  # SC2 (Points not in general position)

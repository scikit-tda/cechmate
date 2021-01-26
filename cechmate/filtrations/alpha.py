import itertools
import numpy as np
import time
import warnings
from numba import njit
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

        if self.verbose:
            print("Doing spatial.Delaunay triangulation...")
            tic = time.time()

        X = X.astype(np.float_)
        delaunay_faces = np.sort(spatial.Delaunay(X).simplices, axis=1)

        if self.verbose:
            print(
                "Finished spatial.Delaunay triangulation (Elapsed Time %.3g)"
                % (time.time() - tic)
            )
            print("Building alpha filtration...")
            tic = time.time()

        self.simplices_ = alpha_build(X, delaunay_faces)

        if self.verbose:
            print(
                "Finished building alpha filtration (Elapsed Time %.3g)"
                % (time.time() - tic)
            )

        return self.simplices_


def alpha_build(X, delaunay_faces):
    """
    Do the Alpha filtration of a Euclidean point set (requires scipy)

    Parameters
    ===========
    X: Nxd array
        Array of N Euclidean vectors in d dimensions
    """
    D = delaunay_faces.shape[1] - 1  # Top dimension
    delaunay_faces = [tuple(simplex) for simplex in delaunay_faces]
    filtration = {dim: {} for dim in range(D, 0, -1)}

    # Special iteration for highest dimensional simplices
    for sigma in delaunay_faces:
        filtration[D][sigma] = _squared_circumradius(X[sigma, :])
        for i in range(D + 1):
            tau = sigma[:i] + sigma[i + 1:]
            if tau in filtration[D - 1] and \
                    not np.isnan(filtration[D - 1][tau]):
                filtration[D - 1][tau] = \
                    min(filtration[D - 1][tau], filtration[D][sigma])
            else:
                x, r_sq = _circumcircle(X[tau, :])
                if np.sum((X[sigma[i]] - x) ** 2) < r_sq:
                    filtration[D - 1][tau] = filtration[D][sigma]
                else:
                    filtration[D - 1][tau] = np.nan

    for dim in range(D - 1, 1, -1):
        for sigma in filtration[dim]:
            if np.isnan(filtration[dim][sigma]):
                filtration[dim][sigma] = _squared_circumradius(X[sigma, :])
            for i in range(dim + 1):
                tau = sigma[:i] + sigma[i + 1:]
                if tau in filtration[dim - 1] and \
                        not np.isnan(filtration[dim - 1][tau]):
                    filtration[dim - 1][tau] = \
                        min(filtration[dim - 1][tau], filtration[dim][sigma])
                else:
                    x, r_sq = _circumcircle(X[tau, :])
                    if np.sum((X[sigma[i]] - x) ** 2) < r_sq:
                        filtration[dim - 1][tau] = filtration[dim][sigma]
                    else:
                        filtration[dim - 1][tau] = np.nan

    # Special iteration for dimension one simplices
    for sigma in filtration[1]:
        if np.isnan(filtration[1][sigma]):
            filtration[1][sigma] = _squared_circumradius(X[sigma, :])

    # Take care of numerical artifacts that may result in simplices with
    # greater filtration values than their co-faces
    for dim in range(D, 1, -1):
        for sigma in filtration[dim]:
            for i in range(dim + 1):
                tau = sigma[:i] + sigma[i + 1:]
                if filtration[dim - 1][tau] > filtration[dim][sigma]:
                    filtration[dim - 1][tau] = filtration[dim][sigma]

    # Convert from squared radii to radii and return list of simplices
    simplices = [((i,), 0) for i in range(X.shape[0])]
    simplices += [
        (sigma, np.sqrt(filtration[dim][sigma]))
        for dim in range(1, D + 1) for sigma in filtration[dim]
        ]

    return simplices


@njit
def _squared_circumradius(X):
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
    if X.shape[0] == 2:
        # Special case of an edge, which is very simple
        dX = X[1] - X[0]
        r_sq = 0.25 * np.sum(dX ** 2)

    else:
        cayleigh_menger = np.ones((X.shape[0] + 1, X.shape[0] + 1))
        cayleigh_menger[0, 0] = 0
        # cayleigh_menger[1:, 1:] = spatial.distance.squareform(
        #     spatial.distance.pdist(X, metric="sqeuclidean")
        #     )
        # cayleigh_menger[1:, 1:] = spatial.distance.cdist(X, X,
        #                                                  metric="sqeuclidean")
        cayleigh_menger[1:, 1:] = _pdist(X)
        bar_coords = -2 * np.linalg.inv(cayleigh_menger)[:1, :]
        r_sq = 0.25 * bar_coords[0, 0]

    return r_sq


@njit
def _circumcircle(X):
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
    if X.shape[0] == 2:
        # Special case of an edge, which is very simple
        dX = X[1] - X[0]
        r_sq = 0.25 * np.sum(dX ** 2)
        x = X[0] + 0.5 * dX

    else:
        cayleigh_menger = np.ones((X.shape[0] + 1, X.shape[0] + 1))
        cayleigh_menger[0, 0] = 0
        # cayleigh_menger[1:, 1:] = spatial.distance.squareform(
        #     spatial.distance.pdist(X, metric="sqeuclidean")
        #     )
        # cayleigh_menger[1:, 1:] = spatial.distance.cdist(X, X,
        #                                                  metric="sqeuclidean")
        cayleigh_menger[1:, 1:] = _pdist(X)
        bar_coords = -2 * np.linalg.inv(cayleigh_menger)[:1, :]
        r_sq = 0.25 * bar_coords[0, 0]
        x = np.sum((bar_coords[:, 1:] / np.sum(bar_coords[:, 1:])) * X.T,
                   axis=1)

    return x, r_sq


@njit
def _pdist(A):
    dist = np.dot(A, A.T)

    TMP = np.empty(A.shape[0], dtype=A.dtype)
    for i in range(A.shape[0]):
        sum = 0.
        for j in range(A.shape[1]):
            sum += A[i, j]**2
        TMP[i] = sum

    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            dist[i, j] *= -2.
            dist[i, j] += TMP[i] + TMP[j]

    return dist

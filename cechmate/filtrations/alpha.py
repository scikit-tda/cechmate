import itertools
import time
import warnings

import numpy as np
from numpy import linalg
from scipy import spatial

from .base import BaseFiltration

__all__ = ["Alpha"]


MIN_DET = 1e-10


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

        ## Step 1: Figure out the filtration
        if self.verbose:
            print("Doing spatial.Delaunay triangulation...")
            tic = time.time()

        delaunay_faces = np.sort(spatial.Delaunay(X).simplices, axis=1)

        if self.verbose:
            print(
                "Finished spatial.Delaunay triangulation (Elapsed Time %.3g)"
                % (time.time() - tic)
            )
            print("Building alpha filtration...")
            tic = time.time()

        filtration = alpha_build(X, delaunay_faces)

        if self.verbose:
            print(
                "Finished building alpha filtration (Elapsed Time %.3g)"
                % (time.time() - tic)
            )

        simplices = [((i,), 0) for i in range(X.shape[0])]
        simplices.extend(filtration.items())

        self.simplices_ = simplices

        return simplices


def alpha_build(X, delaunay_faces):
    """
    Do the Alpha filtration of a Euclidean point set (requires scipy)

    Parameters
    ===========
    X: Nxd array
        Array of N Euclidean vectors in d dimensions
    """
    n_simplices = delaunay_faces.shape[0]
    ambient_dim = delaunay_faces.shape[1] - 1

    # Special iteration for highest order simplices
    highest_simplices_vals = _highest(n_simplices, delaunay_faces, X)
    delaunay_faces = [tuple(simplex) for simplex in delaunay_faces]
    filtration = {delaunay_faces[n]: [highest_simplices_vals[n], True]
                  for n in range(n_simplices)}
    for simplex in delaunay_faces:
        for i in range(ambient_dim + 1):
            tau = simplex[:i] + simplex[i + 1:]
            if tau in filtration:
                filtration[tau] = [min(filtration[tau][0],
                                       filtration[simplex][0]),
                                   False]
            else:
                x, r_sq = _get_circumcircle(X[tau, :])
                if np.sum((X[simplex[i]] - x) ** 2) < r_sq:
                    filtration[tau] = [filtration[simplex][0], False]

    for n_vertices in range(ambient_dim, 2, -1):
        index_combs = list(itertools.combinations(range(ambient_dim + 1),
                                                  n_vertices))
        for simplex in delaunay_faces:
            for idxs in index_combs:
                sigma = tuple([simplex[i] for i in idxs])
                if sigma not in filtration:
                    filtration[sigma] = \
                        [_get_squared_circumradius(X[sigma, :]), False]
                elif filtration[sigma][1]:
                    continue
                if not filtration[sigma][1]:
                    for i in range(n_vertices):
                        tau = sigma[:i] + sigma[i + 1:]
                        if tau in filtration:
                            filtration[tau] = [min(filtration[tau][0],
                                                   filtration[sigma][0]),
                                               False]
                        else:
                            x, r_sq = _get_circumcircle(X[tau, :])
                            if np.sum((X[sigma[i]] - x) ** 2) < r_sq:
                                filtration[tau] = [filtration[sigma][0],
                                                   False]
                    filtration[sigma][1] = True

    index_combs = list(itertools.combinations(range(ambient_dim + 1), 2))
    for simplex in delaunay_faces:
        for idxs in index_combs:
            sigma = tuple([simplex[i] for i in idxs])
            if sigma not in filtration:
                filtration[sigma] = \
                    [_get_squared_circumradius(X[sigma, :]), True]

    # Convert from squared radii to radii
    filtration = {sigma: np.sqrt(filtration[sigma][0]) for sigma in filtration}

    ## Step 2: Take care of numerical artifacts that may result
    ## in simplices with greater filtration values than their co-faces
    simplices_bydim = [set([]) for _ in range(ambient_dim + 2)]
    for simplex in filtration.keys():
        simplices_bydim[len(simplex) - 1].add(simplex)
    simplices_bydim = simplices_bydim[2:]
    simplices_bydim.reverse()
    for simplices_dim in simplices_bydim:
        for sigma in simplices_dim:
            for i in range(len(sigma)):
                tau = sigma[:i] + sigma[i + 1:]
                if filtration[tau] > filtration[sigma]:
                    filtration[tau] = filtration[sigma]

    return filtration


def _highest(n_simplices, delaunay_faces, X):
    values = np.zeros(n_simplices, dtype=np.float_)
    for n in range(n_simplices):
        values[n] = _get_squared_circumradius(X[delaunay_faces[n]])
    return values


def _get_squared_circumradius(X):
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
        cayleigh_menger[1:, 1:] = spatial.distance.squareform(
            spatial.distance.pdist(X, metric="sqeuclidean")
            )
        bar_coords = -2 * np.linalg.inv(cayleigh_menger)[:1, :]
        r_sq = 0.25 * bar_coords[0, 0]

    return r_sq


def _get_circumcircle(X):
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
        cayleigh_menger[1:, 1:] = spatial.distance.squareform(
            spatial.distance.pdist(X, metric="sqeuclidean")
            )
        bar_coords = -2 * np.linalg.inv(cayleigh_menger)[:1, :]
        r_sq = 0.25 * bar_coords[0, 0]
        x = np.sum((bar_coords[:, 1:] / np.sum(bar_coords[:, 1:])) * X.T,
                   axis=1)

    return x, r_sq

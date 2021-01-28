import time
import warnings

import numpy as np
from numba import njit
from numba import types
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.typed import Dict
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

        X = X.astype(np.float64)
        D = X.shape[1]  # Top dimension

        # Need to create dummy NaN dictionaries for Numba's type inference to
        # work.
        filtration_upper = Dict.empty(types.UniTuple(types.int64, D + 2),
                                      types.float64)
        filtration_current = Dict.empty(types.UniTuple(types.int64, D + 1),
                                        types.float64)
        for simplex in np.sort(spatial.Delaunay(X).simplices,
                               axis=1).astype(np.int64):
            filtration_current[tuple(simplex)] = np.nan

        if self.verbose:
            print(
                "Finished spatial.Delaunay triangulation (Elapsed Time %.3g)"
                % (time.time() - tic)
            )
            print("Building alpha filtration...")
            tic = time.time()

        self.simplices_ = []
        for dim in range(D, 0, -1):
            filtration_lower = alpha_build(X, filtration_current,
                                           filtration_upper)
            typ = types.UniTuple(types.int64, dim + 1)
            self.simplices_.extend(
                _unpack_dict_and_sqrt(filtration_current, typ)
                )
            filtration_upper = filtration_current
            filtration_current = filtration_lower

        # Add 0-dimensional simplices
        self.simplices_.extend((((i,), 0.)
                                for i in range(X.shape[0])))

        if self.verbose:
            print(
                "Finished building alpha filtration (Elapsed Time %.3g)"
                % (time.time() - tic)
            )

        return self.simplices_


@njit
def _unpack_dict_and_sqrt(d, typ):
    # Force type inference for `d` given `typ`
    d_dummy = Dict.empty(typ, types.float64)
    key_dummy = next(iter(d))
    d_dummy[key_dummy] = d[key_dummy]

    unpacked_and_sqrt = []
    for key in d:
        unpacked_and_sqrt.append((key, np.sqrt(d[key])))

    return unpacked_and_sqrt


@njit
def alpha_build(X, filtration_current, filtration_upper):
    """
    Do the Alpha filtration of a Euclidean point set (requires scipy)

    Parameters
    ===========
    X: Nxd array
        Array of N Euclidean vectors in d dimensions
    """
    filtration_lower = {}

    if len(next(iter(filtration_current))) == X.shape[1] + 1:
        # Special iteration for highest dimensional simplices, does not check
        # for np.nan
        for sigma in filtration_current:
            filtration_current[sigma] = \
                _squared_circumradius(X[np.asarray(sigma)])
            for x in _drop_elements(sigma):
                tau = x[1]
                vertex = x[0]
                if tau in filtration_lower and \
                        not np.isnan(filtration_lower[tau]):
                    filtration_lower[tau] = \
                        min(filtration_lower[tau], filtration_current[sigma])
                else:
                    x, r_sq = _circumcircle(X[np.asarray(tau)])
                    if np.sum((X[vertex] - x) ** 2) < r_sq:
                        filtration_lower[tau] = filtration_current[sigma]
                    else:
                        filtration_lower[tau] = np.nan

    elif len(next(iter(filtration_current))) == 2:
        # Special iteration for dimension one simplices, does not return
        # filtration_lower
        for sigma in filtration_current:
            if np.isnan(filtration_current[sigma]):
                filtration_current[sigma] = \
                    _squared_circumradius(X[np.asarray(sigma)])

    else:
        for sigma in filtration_current:
            if np.isnan(filtration_current[sigma]):
                filtration_current[sigma] = \
                    _squared_circumradius(X[np.asarray(sigma)])
            for x in _drop_elements(sigma):
                tau = x[1]
                vertex = x[0]
                if tau in filtration_lower and \
                        not np.isnan(filtration_lower[tau]):
                    filtration_lower[tau] = \
                        min(filtration_lower[tau], filtration_current[sigma])
                else:
                    x, r_sq = _circumcircle(X[np.asarray(tau)])
                    if np.sum((X[vertex] - x) ** 2) < r_sq:
                        filtration_lower[tau] = filtration_current[sigma]
                    else:
                        filtration_lower[tau] = np.nan

    # Correct artifacts
    for omega in filtration_upper:
        for x in _drop_elements(omega):
            sigma = x[1]
            filtration_current[sigma] = min(filtration_current[sigma],
                                            filtration_upper[omega])

    return filtration_lower


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


@njit
def _drop_elements(tup: tuple):
    for x in range(len(tup)):
        empty = tup[:-1]  # Not empty, but the right size and will be mutated
        idx = 0
        for i in range(len(tup)):
            if i != x:
                empty = tuple_setitem(empty, idx, tup[i])
                idx += 1
        yield tup[x], empty
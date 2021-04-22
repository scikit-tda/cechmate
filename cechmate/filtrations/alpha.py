import time
import warnings
from functools import lru_cache

import numpy as np
from numba import from_dtype
from numba import njit
from numba import types
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.unsafe.ndarray import to_fixed_tuple
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

    def build(self, X, max_cond=1e3):
        """
        Do the Alpha filtration of a Euclidean point set (requires scipy)

        Parameters
        ===========
        X: Nxd array
            Array of N Euclidean vectors in d dimensions
        """

        if X.shape[0] < X.shape[1]:
            warnings.warn("The input point cloud has more columns than rows; "
                          "did you mean to transpose?")

        if self.verbose:
            print("Doing spatial.Delaunay triangulation...")
            tic = time.time()

        X = X.astype(np.float64)
        D = X.shape[1]  # Top dimension

        delaunay = np.sort(spatial.Delaunay(X).simplices, axis=1)
        idx_dtype = from_dtype(delaunay.dtype)
        alpha_build_top = _alpha_build_top(D)

        if self.verbose:
            print("Finished spatial.Delaunay triangulation (Elapsed Time %.3g)"
                  % (time.time() - tic))
            print("Building alpha filtration...")
            tic = time.time()

        self.simplices_ = []
        filtration_current, filtration_lower = alpha_build_top(X, delaunay, max_cond)
        typ = types.UniTuple(idx_dtype, D + 1)
        self.simplices_.extend(_dict_to_list_sqrt(filtration_current, typ))
        filtration_upper = filtration_current
        filtration_current = filtration_lower

        for dim in range(D - 1, 1, -1):
            filtration_lower = _alpha_build_mid(X, filtration_current,
                                                filtration_upper, max_cond)
            typ = types.UniTuple(idx_dtype, dim + 1)
            self.simplices_.extend(_dict_to_list_sqrt(filtration_current, typ))
            filtration_upper = filtration_current
            filtration_current = filtration_lower

        _alpha_build_bottom(X, filtration_current, filtration_upper, max_cond)
        typ = types.UniTuple(idx_dtype, 2)
        self.simplices_.extend(_dict_to_list_sqrt(filtration_current, typ))

        # Add 0-dimensional simplices
        self.simplices_.extend((((i,), 0.)
                                for i in range(X.shape[0])))

        if self.verbose:
            print("Finished building alpha filtration (Elapsed Time %.3g)"
                  % (time.time() - tic))

        return self.simplices_


@njit
def _dict_to_list_sqrt(d, typ):
    # Force type inference for `d` given `typ`
    # TODO Find a more elegant solution
    d_dummy = Dict.empty(typ, types.float64)
    key_dummy = next(iter(d))
    d_dummy[key_dummy] = d[key_dummy]

    filtration_sqrt = []
    for key in d:
        filtration_sqrt.append((key, np.sqrt(d[key])))

    return filtration_sqrt


@lru_cache
def _alpha_build_top(D):
    len_tups = D + 1  # This needs to be a constant for the inner function

    @njit
    def _alpha_build_top_inner(X, delaunay, max_cond):
        squared_circumradius_func = _squared_circumradius
        circumcircle_func = _circumcircle_edge if len_tups == 3 else _circumcircle

        filtration_current = {}
        filtration_lower = {}

        for sigma_arr in delaunay:
            sigma = to_fixed_tuple(sigma_arr, len_tups)
            filtration_current[sigma] = squared_circumradius_func(X[sigma_arr], max_cond)
            for x in _drop_elements(sigma):
                tau = x[1]
                vertex = x[0]
                if tau in filtration_lower and \
                        not np.isnan(filtration_lower[tau]):
                    filtration_lower[tau] = \
                        min(filtration_lower[tau], filtration_current[sigma])
                else:
                    x, r_sq = circumcircle_func(X[np.asarray(tau)], max_cond)
                    if np.sum((X[vertex] - x) ** 2) < r_sq:
                        filtration_lower[tau] = filtration_current[sigma]
                    else:
                        filtration_lower[tau] = np.nan

        return filtration_current, filtration_lower

    return _alpha_build_top_inner


@njit
def _alpha_build_bottom(X, filtration_current, filtration_upper, max_cond):
    for omega in filtration_upper:
        for x in _drop_elements(omega):
            sigma = x[1]
            if np.isnan(filtration_current[sigma]):
                filtration_current[sigma] = \
                    _squared_circumradius_edge(X[np.asarray(sigma)], max_cond)
            filtration_current[sigma] = min(filtration_current[sigma],
                                            filtration_upper[omega])


@njit
def _alpha_build_mid(X, filtration_current, filtration_upper, max_cond):
    squared_circumradius_func = _squared_circumradius
    circumcircle_func = _circumcircle_edge if len(next(iter(filtration_current))) == 3 else _circumcircle

    filtration_lower = {}

    for sigma in filtration_current:
        if np.isnan(filtration_current[sigma]):
            filtration_current[sigma] = \
                squared_circumradius_func(X[np.asarray(sigma)], max_cond)
        for x in _drop_elements(sigma):
            tau = x[1]
            vertex = x[0]
            if tau in filtration_lower and \
                    not np.isnan(filtration_lower[tau]):
                filtration_lower[tau] = \
                    min(filtration_lower[tau], filtration_current[sigma])
            else:
                x, r_sq = circumcircle_func(X[np.asarray(tau)], max_cond)
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
def _squared_circumradius(X, max_cond):
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
    cayleigh_menger = np.ones((X.shape[0] + 1, X.shape[0] + 1))
    cayleigh_menger[0, 0] = 0
    cayleigh_menger[1:, 1:] = _pdist_sq(X)
    cond = np.linalg.cond(cayleigh_menger)
    if cond > max_cond:
        return np.inf
    bar_coords = -2 * np.linalg.inv(cayleigh_menger)[:1, :]
    r_sq = 0.25 * bar_coords[0, 0]

    return r_sq


@njit
def _squared_circumradius_edge(X, max_cond):
    # Special case of an edge, which is very simple
    dX = X[1] - X[0]
    r_sq = 0.25 * np.sum(dX ** 2)

    return r_sq


@njit
def _circumcircle(X, max_cond):
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
    cayleigh_menger = np.ones((X.shape[0] + 1, X.shape[0] + 1))
    cayleigh_menger[0, 0] = 0
    cayleigh_menger[1:, 1:] = _pdist_sq(X)
    cond = np.linalg.cond(cayleigh_menger)
    if cond > max_cond:
        return np.full(X.shape[1], np.inf), np.inf

    bar_coords = -2 * np.linalg.inv(cayleigh_menger)[:1, :]
    r_sq = 0.25 * bar_coords[0, 0]
    x = np.sum((bar_coords[:, 1:] / np.sum(bar_coords[:, 1:])) * X.T,
               axis=1)

    return x, r_sq


@njit
def _circumcircle_edge(X, max_cond):
    # Special case of an edge, which is very simple
    dX = X[1] - X[0]
    r_sq = 0.25 * np.sum(dX ** 2)
    x = X[0] + 0.5 * dX

    return x, r_sq


@njit
def _pdist_sq(A):
    dist_sq = np.dot(A, A.T)

    TMP = np.empty(A.shape[0], dtype=A.dtype)
    for i in range(A.shape[0]):
        sum = 0.
        for j in range(A.shape[1]):
            sum += A[i, j] ** 2
        TMP[i] = sum

    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            dist_sq[i, j] *= -2.
            dist_sq[i, j] += TMP[i] + TMP[j]
        dist_sq[i, i] = 0.

    return dist_sq


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

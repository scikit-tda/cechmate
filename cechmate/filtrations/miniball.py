#
# This code was adapted from miniball v1.0.2 (https://github.com/marmakoide/miniball)
# to accommodate lru_caching
#
# Modifications contained herein are copyright under same MIT license
# Modifications contained herein under Copyright (c) 2019 Nathaniel Saul
#
# Original Copyright and license:
# Copyright (c) 2019 Alexandre Devert
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np
import random
import functools


def get_circumsphere(S):
    """
    Computes the circumsphere of a set of points
    Parameters
    ----------
    S : (M, N) ndarray, where 1 <= M <= N + 1
        The input points
    Returns
    -------
    C, r2 : ((2) ndarray, float)
        The center and the squared radius of the circumsphere 
    """

    U = S[1:] - S[0]
    B = np.sqrt(np.sum(U ** 2, axis=1))
    U /= B[:, None]
    C = np.dot(np.linalg.solve(np.inner(U, U), 0.5 * B), U)
    return C + S[0], np.sum(C ** 2)


def circle_contains(D, p):
    c, r2 = D
    return np.sum((p - c) ** 2) <= r2


def get_boundary(data, v):
    if len(v) == 0:
        return np.zeros(data.shape[1]), 0.0

    if len(v) <= data.shape[1] + 1:
        return get_circumsphere(data[v])

    c, r2 = get_circumsphere(data[v[: data.shape[1] + 1]])

    # TODO: epsilon is not defined, so not sure how this ever worked?
    # if np.all(np.fabs(np.sum((data[v] - c) ** 2, axis = 1) - r2) < epsilon):
    return c, r2


def miniball_cache(data):
    """ This miniball function is exposed so that the cache can be maintained 
        between subsequent calls. 
        
        Please see the included `miniball` function to see how the interface should be used.
    """

    @functools.lru_cache(maxsize=1000)
    def miniball_rec(tau, v):
        # don't modify tau and v
        tau, v = list(tau), list(v)

        if len(tau) == 0:
            C, r2 = get_boundary(data, v)
        else:
            u = tau.pop()
            C, r2 = miniball_rec(frozenset(tau), frozenset(v))
            if not circle_contains((C, r2), data[u]):
                C, r2 = miniball_rec(frozenset(tau), frozenset(v + [u]))

        return C, r2

    return miniball_rec


def miniball(data):
    """ Miniball algorithm with no caching between runs

    """
    mb = miniball_cache(data)

    C, r2 = mb(frozenset(list(range(data.shape[0]))), frozenset([]))
    return C, np.sqrt(r2)

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

from mock import patch

import numpy as np

import cechmate
from cechmate.filtrations.miniball import miniball_cache, miniball


@patch('cechmate.filtrations.miniball.get_boundary')
def test_caching(mock_get_boundary):

	mock_get_boundary.side_effect = cechmate.filtrations.get_boundary
	points = np.array([
		[0.0,0.1], [0.5, 0.5], [0.0, 1.0], [0.7, 0.7]
	])

	mb = miniball_cache(points)
	C, r = mb(frozenset(list(range(3))), frozenset([]))
	
	# import pdb; pdb.set_trace()
	count = mock_get_boundary.call_count
	assert count == 8
	# compute a subset
	C, r = mb(frozenset(list(range(2))), frozenset([]))
	assert mock_get_boundary.call_count == count

	# compute a superset
	C, r = mb(frozenset(list(range(4))), frozenset([]))
	assert mock_get_boundary.call_count == 12

def test_vertex():
	points = np.array([
		[2.45, 0.5]
	])
	C, r = miniball(points)
	assert r == 0.0
	assert np.array_equal(C, points[0])

def test_simple_case():
	points = np.array([
		[0.0,0.0], [0.0, 0.5], [0.0, 1.0]
	])

	C, r = miniball(points)
	assert r == 0.5
 
def test_bounding_ball_contains_point_set():
	# Check that the computed bounding ball contains all the input points
	for n in range(1, 10):
		for count in range(2, n + 10):
			# Generate points
			S = np.random.randn(count, n)

			# Get the bounding sphere
			C, r2 = miniball(S)

			# Check that all points are inside the bounding sphere up to machine precision
			assert np.all(np.sum((S - C) ** 2, axis = 1) - r2**2 < 1e-12)



def test_bounding_ball_optimality():
	# Check that the bounding ball are optimal
	for n in range(2, 10):
		for count in range(n + 2, n + 30):
			# Generate a support sphere from n+1 points
			S_support = np.random.randn(n + 1, n)
			C_support, r2_support = miniball(S_support)
			
			# Generate points inside the support sphere
			S = np.random.randn(count - S_support.shape[0], n)
			S /= np.sqrt(np.sum(S ** 2, axis = 1))[:,None]
			S *= ((.9 * r2_support) * np.random.rand(count - S_support.shape[0], 1))
			S = S + C_support
				
			# Get the bounding sphere
			C, r2 = miniball(np.concatenate([S, S_support], axis = 0))

			# Check that the bounding sphere and the support sphere are equivalent
			# up to machine precision.
			assert np.allclose(r2, r2_support)
			assert np.allclose(C, C_support)


import pytest

import numpy as np
from cechmate.filtrations import Rips


@pytest.fixture
def two_points():
    x = np.array([[0, 0.0], [1, 1.0]])

    return x


def test_two_points(two_points):
    r = Rips(maxdim=2).fit(two_points)

    assert len(r) == 3

    vertices = [s for s in r if len(s[0]) == 1]
    edges = [s for s in r if len(s[0]) == 2]

    assert len(vertices) == 2
    assert len(edges) == 1


def test_correct_edge_length(two_points):
    r = Rips(maxdim=2).fit(two_points)

    vertices = [s for s in r if len(s[0]) == 1]
    edges = [s for s in r if len(s[0]) == 2]

    assert vertices[0][1] == 0.0
    assert edges[0][1] == np.sqrt(2)


@pytest.fixture
def equilateral_triangle():
    """Define an equilateral triangle to see the difference between Cech and Rips."""
    x = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    return x


def test_triangle(equilateral_triangle):
    """Expect 3 vertices, 3 edges, and a triangle."""
    r = Rips(maxdim=2)
    r_simplices = r.fit(equilateral_triangle)

    assert len(r_simplices) == 7

    vertices = [s for s in r_simplices if len(s[0]) == 1]
    edges = [s for s in r_simplices if len(s[0]) == 2]
    triangles = [s for s in r_simplices if len(s[0]) == 3]

    assert len(vertices) == 3
    assert len(edges) == 3
    assert len(triangles) == 1

    r_diagrams = r.transform(r_simplices)
    assert len(r_diagrams) == 1
    assert len(r_diagrams[0]) == 3

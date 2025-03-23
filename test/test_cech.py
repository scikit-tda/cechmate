import pytest

import numpy as np
from cechmate.filtrations import Cech



@pytest.fixture
def equilateral_triangle():
    """Define an equilateral triangle to see importance of Cech."""
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
    c = Cech(maxdim=2)
    c_simplices = c.fit(equilateral_triangle)

    assert len(c_simplices) == 7

    vertices = [s for s in c_simplices if len(s[0]) == 1]
    edges = [s for s in c_simplices if len(s[0]) == 2]
    triangles = [s for s in c_simplices if len(s[0]) == 3]

    assert len(vertices) == 3
    assert len(edges) == 3
    assert len(triangles) == 1

    c_diagrams = c.transform(c_simplices)
    assert len(c_diagrams) == 2
    assert len(c_diagrams[0]) == 3
    assert len(c_diagrams[1]) == 1

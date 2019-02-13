import pytest

import numpy as np
from cechmate import Cech

@pytest.fixture
def triangle():
    x = np.array([
        [0, 0.0], 
        [1, 1.0],
        [0, 1.0],
    ])

    return x

def test_triangle(triangle):
    """ Expect 3 vertices, 3 edges, and a triangle

    """
    c = Cech(2).build(triangle)

    assert len(c) == 7 

    vertices = [s for s in c if len(s[0]) == 1]
    edges = [s for s in c if len(s[0]) == 2]
    triangles = [s for s in c if len(s[0]) == 3]

    assert len(vertices) == 3
    assert len(edges) == 3
    assert len(triangles) == 1



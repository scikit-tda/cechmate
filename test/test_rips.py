import pytest

import numpy as np
from cechmate import Rips


@pytest.fixture
def two_points():
    x = np.array([[0, 0.0], [1, 1.0]])

    return x


def test_two_points(two_points):
    r = Rips(2).build(two_points)

    assert len(r) == 3

    vertices = [s for s in r if len(s[0]) == 1]
    edges = [s for s in r if len(s[0]) == 2]

    assert len(vertices) == 2
    assert len(edges) == 1


def test_correct_edge_length(two_points):
    r = Rips(2).build(two_points)

    vertices = [s for s in r if len(s[0]) == 1]
    edges = [s for s in r if len(s[0]) == 2]

    assert vertices[0][1] == 0.0
    assert edges[0][1] == np.sqrt(2)

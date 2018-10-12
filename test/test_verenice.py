import pytest

from cechmate import Verenice


@pytest.fixture
def two_cover():
    cover = {1: [0, 1, 2], 2: [0, 1, 3]}
    return cover


def test_single_edge(two_cover):
    v = Verenice(2).build(two_cover)

    assert len(v) == 3


def test_jaccard_dist_equal():
    a = [0, 1]
    b = [0, 1]

    v = Verenice(2)

    assert v.jaccard([a, b]) == 0


def test_jaccard_dist_disjoint():
    a = [0, 1]
    b = [4, 5]
    c = [10, 20]

    v = Verenice(2)

    assert v.jaccard([a, b, c]) == 1


def test_triangles():
    a = [0, 1]
    b = [1, 5]
    c = [1, 6]

    v = Verenice(3).build([a, b, c])

    assert len(v) == 7

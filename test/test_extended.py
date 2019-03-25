import numpy as np
import pytest

import kmapper
import networkx as nx

from cechmate import Extended
from cechmate.filtrations.extended import _lower_star

@pytest.fixture
def reeb():
    """ This example taken from the reeb graph in Carri`ere 2017 of Figure 4"""

    f = {
        1: 0.0,
        2: 0.5,
        3: 1.0,
        4: 1.5,
        5: 1.5,
        6: 2.0,
        7: 2.0,
        8: 2.5, 
        9: 3.0,
        10: 3.5
    }

    X = [
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
        [1,3], [2,4], [3,4], [3,5], [4,6], [5,7], [7,9], [7,8], [6,8], [8,10]
    ]

    expected = {
        0: {
            "ordinary": [[0.5, 1.5]],
            "extended": [[0.0, 3.5]],
        },
        1: {
            "extended": [[2.5, 1.0]],
            "relative": [[3.0, 2.0]]
        }
    }

    return X, f, expected

@pytest.fixture
def X():
    f = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
    }

    X = [
        [0], [1], [2], [3], [4], [5], [6],
        [0,2], [1,3], [2,4], [3,4], [4,5], [4,6]
    ]

    return X, f

@pytest.fixture
def triangle():
    X = [
        [0],[1],[2],
        [0,1],[1,2],[0,2],
        [0,1,2]
    ]
    f = {0:4.1, 1:1.1, 2:0.1}

    return X, f

def test_reeb_known(reeb):
    X, f, expected = reeb

    diagrams = Extended(X, f).diagrams()
    assert expected == diagrams

def test_lower_star(triangle):
    X, f = triangle
    lstar = _lower_star(X, 0, f)

    assert len(lstar) == 4
    assert [0,1,2] in lstar

    f = {0:4, 1:1, 2:1}
    lstar = _lower_star(X, 1, f)

    assert len(lstar) == 2
    assert [0,1,2] not in lstar

def test_lower_boundary_matrix(triangle):
    X, f = triangle

    bm, _ = Extended(X, f)._up_down_boundary_matrix(X, f)
    assert bm == [ # this was manually computed by @sauln
        (0, []), 
        (0, []), 
        (1, [0, 1]), 
        (0, []), 
        (1, [1, 3]), 
        (1, [0, 3]), 
        (2, [2,4,5]),
        (1, [3]), 
        (1, [1]), 
        (2, [4, 7, 8]), 
        (1, [0]), 
        (2, [2, 8, 10]), 
        (2, [5, 7, 10]),
        (3, [6, 9, 11, 12])
    ]

def test_reduction(triangle):
    X, f = triangle
    e = Extended(X, f)
    bm, _ = e._up_down_boundary_matrix(X, f)

    pairs = e._compute_persistence_pairs(bm)
    assert pairs == [
        (0, 7), (1, 2), (3, 4), (5, 6), (8, 9), (10, 11), (12, 13)
    ]



class TestConstructors:
    def test_from_kmapper_simplices(self):
        km = kmapper.KeplerMapper()
        data = np.random.random((300, 5))
        lens = km.project(data)
        graph = km.map(lens, data)

        e = Extended.from_kmapper(graph, lens)

        vs = [s for s in e.simplices if len(s) == 1]
        ls = [s for s in e.simplices if len(s) == 2]
        assert len(vs) == len(graph['nodes'])
        assert len(ls) == sum(len(v) for v in graph['links'].values())
        assert len(e.simplices) == len(graph['simplices'])

    def test_from_kmapper_mapping_lens(self):
        km = kmapper.KeplerMapper()
        np.random.seed(0)
        data = np.random.random((300, 5))
        lens = km.project(data)
        graph = km.map(lens, data)

        e = Extended.from_kmapper(graph, lens)

        assert len(graph['nodes']) == len(e.f)
        assert set(e.f.values()) == set(np.mean(lens[v]) for v in graph['nodes'].values())

    def test_from_kmapper_mapping_nodes(self):
        km = kmapper.KeplerMapper()
        np.random.seed(0)
        data = np.random.random((300, 5))
        lens = km.project(data)
        graph = km.map(lens, data)

        f = {k: np.random.random() for k in graph['nodes']}
        e = Extended.from_kmapper(graph, f)

        assert len(f) == len(e.f)
        assert set(e.f.values()) == set(f.values())

    def test_from_nx_weights(self):
        g = nx.tutte_graph()
        vals = {n: np.random.random() for n in g.nodes}
        nx.set_node_attributes(g, vals, "weight_str")
        e = Extended.from_nx(g, "weight_str")

        assert len(e.simplices) == len(g.nodes) + len(g.edges)
        assert set(vals.values()) == set(e.f.values())

    def test_from_nx_map(self):      
        g = nx.tutte_graph()
        vals = {n: np.random.random() for n in g.nodes}
        e = Extended.from_nx(g, vals)

        assert len(e.simplices) == len(g.nodes) + len(g.edges)
        assert set(vals.values()) == set(e.f.values())

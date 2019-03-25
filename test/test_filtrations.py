import time

import numpy as np

from cechmate import phat_diagrams, Alpha, Rips


def test_phat_diagrams():

    t = np.linspace(0, 2 * np.pi, 40)
    X = np.zeros((len(t), 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    np.random.seed(10)
    X += 0.2 * np.random.randn(len(t), 2)
    rips = Rips(1).build(X)

    dgms = phat_diagrams(rips)


def test_rips():
    """
    A test with a noisy circle, comparing H1 to GUDHI
    """
    t = np.linspace(0, 2 * np.pi, 40)
    X = np.zeros((len(t), 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    np.random.seed(10)
    X += 0.2 * np.random.randn(len(t), 2)
    rips = Rips(1).build(X)


def test_alpha():

    # Make a 3-sphere in 4 dimensions
    X = np.random.randn(15, 4)
    X = X / np.sqrt(np.sum(X ** 2, 1)[:, None])
    tic = time.time()
    diagrams = Alpha().build(X)
    phattime = time.time() - tic

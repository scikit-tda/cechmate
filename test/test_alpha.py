import time

import numpy as np

from cechmate import Alpha


def test_alpha():

    # Make a 3-sphere in 4 dimensions
    X = np.random.randn(15, 4)
    X = X / np.sqrt(np.sum(X ** 2, 1)[:, None])
    tic = time.time()
    diagrams = Alpha().build(X)
    phattime = time.time() - tic


def test_alpha_filtration_is_not_too_large():
    # https://github.com/scikit-tda/cechmate/issues/10
    # The points from the following test set are
    # drawn from a standard normal 2D distribution
    # No pair of points has a distance larger than 4
    # Actually, the largest circumradius should be ~1.8
    points = np.array(
      [[ 0.01743489,  0.83907818],
       [-0.57518843,  0.46536324],
       [-0.19659281, -0.66731467],
       [ 1.52911009, -0.68218385],
       [-0.66838721, -2.21357309],
       [ 1.14180137,  0.79701124],
       [-0.05349503, -2.25566765],
       [-0.27223817,  0.77621451],
       [ 0.38597224,  1.15861246],
       [-0.29454972,  1.71746955],
       [-0.68879532,  1.36314908],
       [-0.47834989, -1.57854915],
       [ 0.94477495, -0.38586968],
       [-0.04377718, -0.84981483],
       [-0.03082609, -1.63861901],
       [ 1.73579262, -0.02458939],
       [ 0.50910058,  0.66446628],
       [ 1.88017434,  1.66114513],
       [ 1.47186944, -0.68486166]])
    simplices, filtration = zip(*Alpha().build(points))
    assert max(filtration) < 2


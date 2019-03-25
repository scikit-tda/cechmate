import numpy as np

from cechmate import phat_diagrams


def test_1d_solver():

    filt = [
        ([0], 0.0),
        ([1], 0.0),
        ([2], 0.0),
        ([3], 0.0),
        ([0, 1], 0.5),
        ([1, 2], 0.6),
    ]

    dgms = phat_diagrams(filt)

    assert len(dgms[0]) == 2
    assert [0, 0.5] in dgms[0].tolist()
    assert [0.0, 0.6] in dgms[0].tolist()


def test_infs():
    filt = [
        ([0], 0.0),
        ([1], 0.0),
        ([2], 0.0),
        ([3], 0.0),
        ([0, 1], 0.5),
        ([1, 2], 0.6),
        ([0, 2], 0.7),
    ]

    dgms = phat_diagrams(filt, show_inf=True)

    assert len(dgms) == 2

    assert [0.7, np.inf] in dgms[1].tolist()

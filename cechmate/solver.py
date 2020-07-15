import itertools
import time

import numpy as np
import phat


def phat_diagrams(simplices, show_inf=False, verbose=True):
    """
    Compute the persistence diagram for :code:`simplices` using Phat.

    Parameters
    -----------
    simplices: A list of lists of simplices and their distances
        the kth element is itself a list of tuples ([idx1, ..., idxk], dist)
        where [idx1, ..., idxk] is a list of vertices involved in the simplex
        and "dist" is the distance at which the simplex is added

    show_inf: Boolean
        Determines whether or not to return points that never die.

    Returns
    --------
    dgms: list of diagrams 
        the persistence diagram for Hk 
    """

    ## Convert simplices representation to sparse pivot column
    #  -- sort by birth time, if tie, use order of simplex
    ordered_simplices = sorted(simplices, key=lambda x: (x[1], len(x[0])))
    columns = _simplices_to_sparse_pivot_column(ordered_simplices, verbose)

    ## Setup boundary matrix and reduce
    if verbose:
        print("Computing persistence pairs...")
        tic = time.time()

    boundary_matrix = phat.boundary_matrix(
        columns=columns, representation=phat.representations.sparse_pivot_column
    )
    pairs = boundary_matrix.compute_persistence_pairs()
    pairs.sort()

    if verbose:
        print(
            "Finished computing persistence pairs (Elapsed Time %.3g)"
            % (time.time() - tic)
        )

    ## Setup persistence diagrams by reading off distances
    dgms = _process_distances(pairs, ordered_simplices)

    ## Add all unpaired simplices as infinite points
    if show_inf:
        dgms = _add_unpaired(dgms, pairs, simplices)

    ## Convert to arrays:
    dgms = [np.array(dgm) for dgm in dgms.values()]

    return dgms


def _simplices_to_sparse_pivot_column(ordered_simplices, verbose=False):
    """

    """

    idx = 0
    columns = []
    idxs2order = {}

    if verbose:
        print("Constructing boundary matrix...")
        tic = time.time()

    for idxs, dist in ordered_simplices:
        k = len(idxs)
        idxs = sorted(idxs)
        idxs2order[tuple(idxs)] = idx
        idxs = np.array(idxs)
        if len(idxs) == 1:
            columns.append((0, []))
        else:
            # Get all faces with k-1 vertices
            collist = []
            for fidxs in itertools.combinations(range(k), k - 1):
                fidxs = np.array(list(fidxs))
                fidxs = tuple(idxs[fidxs])
                if not fidxs in idxs2order:
                    raise Exception(
                        "Error: Not a proper filtration: %s added before %s"
                        % (idxs, fidxs)
                    )
                    return None
                collist.append(idxs2order[fidxs])
            collist = sorted(collist)
            columns.append((k - 1, collist))

        idx += 1

    if verbose:
        print(
            "Finished constructing boundary matrix (Elapsed Time %.3g)"
            % (time.time() - tic)
        )

    return columns


def _process_distances(pairs, ordered_simplices):
    """ Setup persistence diagrams by reading off distances
    """

    dgms = {}
    posneg = np.zeros(len(ordered_simplices))

    for [bi, di] in pairs:
        bidxs, bd = ordered_simplices[bi]
        didxs, dd = ordered_simplices[di]

        assert posneg[bi] == 0 and posneg[di] == 0
        posneg[bi], posneg[di] = 1, -1

        assert dd >= bd
        # assert len(bidxs) == len(didxs) - 1

        p = len(bidxs) - 1

        # Don't add zero persistence pairs
        if bd != dd:
            dgms.setdefault(p, []).append([bd, dd])

    return dgms


def _add_unpaired(dgms, pairs, simplices):
    posneg = np.zeros(len(simplices))
    for [bi, di] in pairs:
        assert posneg[bi] == 0
        assert posneg[di] == 0
        posneg[bi] = 1
        posneg[di] = -1

    for i in range(len(posneg)):
        if posneg[i] == 0:
            (idxs, dist) = simplices[i]
            p = len(idxs) - 1
            if not p in dgms:
                dgms[p] = []
            dgms[p].append([dist, np.inf])

    return dgms

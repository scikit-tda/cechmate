import itertools
import time

import numpy as np
import phat

from .filtrations.simplex import Simplex


def phat_diagrams(simplices, returnInfs=False, verbose=True):
    """
    Do a custom filtration wrapping around phat

    :param simplices: A list of lists of simplices and their distances\
        the kth element is itself a list of tuples ([idx1, ..., idxk], dist)\
        where [idx1, ..., idxk] is a list of vertices involved in the simplex\
        and "dist" is the distance at which the simplex is added
    :param returnInfs: Whether or not to return points that never die
    :param useWrapper: If true, call the phat binary as a subprocess.  If \
        false, use Python bindings
    :returns Is: A dictionary of persistence diagrams, where Is[k] is \
        the persistence diagram for Hk 
    """
    idxs2order = {}
    ## Step 1: Go through simplices in ascending order of distance
    idx = 0
    columns = []
    ordsimplices = sorted([Simplex(s[0], s[1]) for s in simplices])
    if verbose:
        print("Constructing boundary matrix...")
        tic = time.time()
    for simplex in ordsimplices:
        (idxs, dist) = (simplex.idxs, simplex.dist)
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
                    print(
                        "Error: Not a proper filtration: %s added before %s"
                        % (idxs, fidxs)
                    )
                    return None
                collist.append(idxs2order[fidxs])
            collist = sorted(collist)
            columns.append((k - 1, collist))
        idx += 1

    ## Step 2: Setup boundary matrix and reduce
    if verbose:
        print(
            "Finished constructing boundary matrix (Elapsed Time %.3g)"
            % (time.time() - tic)
        )
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

    ## Step 3: Setup persistence diagrams by reading off distances
    Is = {}  # Persistence diagrams
    posneg = np.zeros(len(simplices))
    for [bi, di] in pairs:
        # Distances
        (bidxs, bd) = [ordsimplices[bi].idxs, ordsimplices[bi].dist]
        (didxs, dd) = [ordsimplices[di].idxs, ordsimplices[di].dist]
        assert posneg[bi] == 0
        assert posneg[di] == 0
        posneg[bi] = 1
        posneg[di] = -1
        assert dd >= bd
        assert len(bidxs) == len(didxs) - 1
        p = len(bidxs) - 1
        if not p in Is:
            Is[p] = []
        if bd == dd:
            # Don't add zero persistence pairs
            continue
        Is[p].append([bd, dd])

    ## Step 4: Add all unpaired simplices as infinite points
    if returnInfs:
        for i in range(len(posneg)):
            if posneg[i] == 0:
                (idxs, dist) = simplices[i]
                p = len(idxs) - 1
                if not p in Is:
                    Is[p] = []
                Is[p].append([dist, np.inf])
    for i in range(len(Is)):
        Is[i] = np.array(Is[i])
    return Is

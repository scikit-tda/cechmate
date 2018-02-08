import numpy as np
import matplotlib.pyplot as plt
import itertools

def plotDGM(dgm, color = 'b', sz = 20, label = 'dgm', \
        axcolor = np.array([0.0, 0.0, 0.0]), marker = None):
    """
    Plot a persistence diagram
    :param dgm: An NPoints x 2 array of birth and death times
    :param color: A color for the points (default 'b' for blue)
    :param sz: Size of the points
    :param label: Label to associate with the diagram
    :param axcolor: Color of the diagonal
    :param marker: Type of marker (e.g. 'x' for an x marker)
    :returns H: A handle to the plot
    """
    if dgm.size == 0:
        return
    # Create Lists
    # set axis values
    axMin = np.min(dgm)
    axMax = np.max(dgm)
    axRange = axMax-axMin
    a = max(axMin - axRange/5, 0)
    b = axMax+axRange/5
    # plot line
    plt.plot([a, b], [a, b], c = axcolor, label = 'none')
    # plot points
    if marker:
        H = plt.scatter(dgm[:, 0], dgm[:, 1], sz, color, marker, label=label, edgecolor = 'none')
    else:
        H = plt.scatter(dgm[:, 0], dgm[:, 1], sz, color, label=label, edgecolor = 'none')
    # add labels
    plt.xlabel('Time of Birth')
    plt.ylabel('Time of Death')
    return H

class Simplex(object):
    """
    A class to help sort simplices.  Sorted by distance, and sorted by order
    if tied
    """
    def __init__(self, idxs, dist):
        self.idxs = idxs
        self.dist = dist
    
    def __eq__(self, other):
        return (self.dist == other.dist) and (len(self.idxs) == len(other.idxs))
    
    def __ne__(self, other):
        return not (self.dist == other.dist) or not (len(self.idxs) == len(other.idxs))
    
    def __lt__(self, other):
        if self.dist < other.dist:
            return True
        elif self.dist == other.dist:
            if len(self.idxs) < len(other.idxs):
                return True
        return False
    
    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)
    
    def __gt__(self, other):
        if self.dist > other.dist:
            return True
        elif self.dist == other.dist:
            if len(self.idxs) > len(other.idxs):
                return True
        return False
    
    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)
    
    def __repr__(self):
        return "%s %s"%(self.idxs, self.dist)


def getCechDGMs(simplices, returnInfs = False, useWrapper = True, Verbose = True):
    """
    Do a Cech filtration
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
    #Step 1: Go through simplices in ascending order of distance
    idx = 0
    columns = []
    ordsimplices = sorted([Simplex(s[0], s[1]) for s in simplices])
    for simplex in ordsimplices:
        (idxs, dist) = (simplex.idxs, simplex.dist)
        k = len(idxs)
        idxs = sorted(idxs)
        idxs2order[tuple(idxs)] = idx
        idxs = np.array(idxs)
        if len(idxs) == 1:
            columns.append((0, []))
        else:
            #Get all faces with k-1 vertices
            collist = []
            for fidxs in itertools.combinations(range(k), k-1):
                fidxs = np.array(list(fidxs))
                fidxs = tuple(idxs[fidxs])
                if not fidxs in idxs2order:
                    print("Error: Not a proper filtration: %s added before %s"\
                        %(idxs, fidxs))
                    return None
                collist.append(idxs2order[fidxs])
            columns.append((k-1, collist))
        idx += 1
    #Step 2: Setup boundary matrix and reduce
    if Verbose:
        print("Computing persistence pairs...")
    if useWrapper:
        import subprocess
        fout = open("boundary.dat", "w")
        for (k, collist) in columns:
            fout.write("%i "%k)
            fout.write(("%i "*len(collist))%tuple(collist))
            fout.write("\n")
        fout.close()
        subprocess.call(["./phat", "--ascii", "boundary.dat", "pairs.dat"])
        fin = open("pairs.dat")
        pairs = []
        for line in fin.readlines()[1::]:
            pairs.append([int(k.strip()) for k in line.split()])
        fin.close()
    else:
        import phat
        boundary_matrix = phat.boundary_matrix(columns = columns)
        pairs = boundary_matrix.compute_persistence_pairs()
        pairs.sort()
    if Verbose:
        print("Finished computing persistence pairs")

    #Step 3: Setup persistence diagrams by reading off distances
    Is = {} #Persistence diagrams
    posneg = np.zeros(len(simplices))
    for [bi, di] in pairs:
        #Distances
        (bidxs, bd) = [ordsimplices[bi].idxs, ordsimplices[bi].dist]
        (didxs, dd) = [ordsimplices[di].idxs, ordsimplices[di].dist]
        assert(posneg[bi] == 0)
        assert(posneg[di] == 0)
        posneg[bi] = 1
        posneg[di] = -1
        assert(dd >= bd)
        assert(len(bidxs) == len(didxs)-1)
        p = len(bidxs)-1
        if not p in Is:
            Is[p] = []
        if bd == dd:
            #Don't add zero persistence pairs
            continue
        Is[p].append([bd, dd])
    
    #Step 4: Add all unpaired simplices as infinite points
    if returnInfs:
        for i in range(len(posneg)):
            if posneg[i] == 0:
                (idxs, dist) = simplices[i]
                p = len(idxs)-1
                if not p in Is:
                    Is[p] = []
                Is[p].append([dist, np.inf])
    for i in range(len(Is)):
        Is[i] = np.array(Is[i])
    return Is

def getSSM(X):
    """
    Given a set of Euclidean vectors, return a pairwise distance matrix
    :param X: An Nxd array of N Euclidean vectors in d dimensions
    :returns D: An NxN array of all pairwise distances
    """
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2 * X.dot(X.T)
    D[D < 0] = 0  # Numerical precision
    D = np.sqrt(D)
    return D

def doRipsFiltrationEuclidean(X, p):
    """
    Do the rips filtration of a Euclidean point set
    :param X: An Nxd array of N Euclidean vectors in d dimensions
    :param p: The order of homology to go up to
    :returns Is: A dictionary of persistence diagrams, where Is[k] is \
        the persistence diagram for Hk 
    """
    D = getSSM(X)
    N = D.shape[0]
    xr = np.arange(N)
    xrl = xr.tolist()
    #First add all 0 simplices
    simplices = [([i], 0) for i in range(N)]
    for k in range(p+1):
        #Add all (k+1)-simplices, which have (k+2) vertices
        for idxs in itertools.combinations(xrl, k+2):
            idxs = list(idxs)
            d = 0.0
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    d = max(d, D[idxs[i], idxs[j]])
            simplices.append((idxs, d))
    return getCechDGMs(simplices)    

def doAlphaComplexFiltration(X, p):
    """
    Do the Cech filtration of a Euclidean point set (requires scipy)
    :param X: An Nxd array of N Euclidean vectors in d dimensions
    """
    from scipy.spatial import Delaunay
    pass

def doRipsFiltrationDMGUDHI(D, p, coeff = 2, doPlot = False):
    """
    Do the rips filtration, wrapping around the GUDHI library (for comparison)
    :param X: An Nxk matrix of points
    :param p: The order of homology to go up to
    :param coeff: The field coefficient of homology
    :returns Is: A dictionary of persistence diagrams, where Is[k] is \
        the persistence diagram for Hk 
    """
    import gudhi
    rips = gudhi.RipsComplex(distance_matrix=D,max_edge_length=np.inf)
    simplex_tree = rips.create_simplex_tree(max_dimension=p+1)
    diag = simplex_tree.persistence(homology_coeff_field=coeff, min_persistence=0)
    if doPlot:
        pplot = gudhi.plot_persistence_diagram(diag)
        pplot.show()
    Is = []
    for i in range(p+1):
        Is.append([])
    for (i, (b, d)) in diag:
        Is[i].append([b, d])
    for i in range(len(Is)):
        Is[i] = np.array(Is[i])
    return Is

def testRips(compareToGUDHI = False):
    """
    A test with a noisy circle, comparing H1 to GUDHI
    """
    t = np.linspace(0, 2*np.pi, 100)
    X = np.zeros((len(t), 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    np.random.seed(10)
    X += 0.2*np.random.randn(len(t), 2)
    Is = doRipsFiltrationEuclidean(X, 1)
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], 20)
    plt.axis('equal')
    plt.subplot(132)
    plotDGM(Is[0])
    plt.title("H0")
    plt.subplot(133)
    plotDGM(Is[1])
    if compareToGUDHI:
        D = getSSM(X)
        Is2 = doRipsFiltrationDMGUDHI(D, 1)
        plotDGM(Is2[1], color = 'r', marker = 'x')
    plt.title("H1")
    plt.show()


if __name__ == '__main__':
    testRips()
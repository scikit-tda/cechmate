import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import itertools
import phat
import warnings
import time

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


def get_phat_dgms(simplices, returnInfs = False, Verbose = True):
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
    if Verbose:
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
            collist = sorted(collist)
            columns.append((k-1, collist))
        idx += 1
    ## Step 2: Setup boundary matrix and reduce
    if Verbose:
        print("Finished constructing boundary matrix (Elapsed Time %.3g)"%(time.time()-tic))
        print("Computing persistence pairs...")
        tic = time.time()
    boundary_matrix = phat.boundary_matrix(columns = columns, representation = phat.representations.sparse_pivot_column)
    pairs = boundary_matrix.compute_persistence_pairs()
    pairs.sort()
    if Verbose:
        print("Finished computing persistence pairs (Elapsed Time %.3g)"%(time.time()-tic))

    ## Step 3: Setup persistence diagrams by reading off distances
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
    
    ## Step 4: Add all unpaired simplices as infinite points
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

def rips_filtration(X, p):
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
    return get_phat_dgms(simplices)    

def get_circumcenter(X):
    """
    Compute the circumcenter and circumradius of a simplex
    Parameters
    ----------
    X : ndarray (N, d)
        Coordinates of points on an N-simplex in d dimensions
    
    Returns
    -------
    (circumcenter, circumradius)
        A tuple of the circumcenter and squared circumradius.  
        (SC1) If there are fewer points than the ambient dimension plus one,
          then return the circumcenter corresponding to the smallest
          possible squared circumradius
        (SC2) If the points are not in general position, 
          it returns (np.inf, np.inf)
        (SC3) If there are more points than the ambient dimension plus one
          it returns (np.nan, np.nan)
    """
    if X.shape[0] == 2:
        # Special case of an edge, which is very simple
        dX = X[1, :] - X[0, :]
        rSqr = 0.25*np.sum(dX**2)
        x = X[0, :] + 0.5*dX
        return (x, rSqr)
    if X.shape[0] > X.shape[1] + 1: # SC3 (too many points)
        warnings.warn("Trying to compute circumsphere for " +\
                      "%i points in %i dimensions"%(X.shape[0], X.shape[1]))
        return (np.nan, np.nan)
    # Transform arrays for PCA for SC1 (points in higher ambient dimension)
    muV = np.array([])
    V = np.array([]) 
    if X.shape[0] < X.shape[1]+1:
        # SC1: Do PCA down to NPoints-1
        muV = np.mean(X, 0)
        XCenter = X - muV
        _, V = linalg.eigh((XCenter.T).dot(XCenter))
        V = V[:, (X.shape[1]-X.shape[0]+1)::] #Put dimension NPoints-1
        X = XCenter.dot(V)
    muX = np.mean(X, 0)
    D = np.ones((X.shape[0], X.shape[0]+1))
    # Subtract off centroid for numerical stability
    D[:, 1:-1] = X - muX
    D[:, 0] = np.sum(D[:, 1:-1]**2, 1)
    minor = lambda A, j: \
            A[:, np.concatenate((np.arange(j), np.arange(j+1, A.shape[1])))]
    dxs = np.array([linalg.det(minor(D, i)) for i in range(1, D.shape[1]-1)])
    alpha = linalg.det(minor(D, 0))
    if np.abs(alpha) > 0:
        signs = (-1)**np.arange(len(dxs))
        x = dxs*signs/(2*alpha) + muX #Add back centroid
        gamma = ((-1)**len(dxs))*linalg.det(minor(D, D.shape[1]-1))
        rSqr = (np.sum(dxs**2) + 4*alpha*gamma)/(4*alpha*alpha)
        if V.size > 0:
            # Transform back to ambient if SC1
            x = x.dot(V.T) + muV
        return (x, rSqr)
    return (np.inf, np.inf) #SC2 (Points not in general position)

def alpha_filtration(X, Verbose = True):
    """
    Do the Alpha filtration of a Euclidean point set (requires scipy)
    :param X: An Nxd array of N Euclidean vectors in d dimensions
    """
    from scipy.spatial import Delaunay

    if X.shape[0] < X.shape[1]:
        warnings.warn(
            "The input point cloud has more columns than rows; " +
            "did you mean to transpose?")
    maxdim = X.shape[1]-1

    ## Step 1: Figure out the filtration
    if Verbose:
        print("Doing Delaunay triangulation...")
        tic = time.time()
    delaunay_faces = Delaunay(X).simplices
    if Verbose:
        print("Finished Delaunay triangulation (Elapsed Time %.3g)"%(time.time()-tic))
        print("Building alpha filtration...")
        tic = time.time()
        
    filtration = {}
    simplices_bydim = {}
    for dim in range(maxdim+2, 1, -1):
        simplices_bydim[dim] = []
        for s in range(delaunay_faces.shape[0]):
            simplex = delaunay_faces[s, :]
            for sigma in itertools.combinations(simplex, dim):
                sigma = tuple(sorted(sigma))
                simplices_bydim[dim].append(sigma)
                if not sigma in filtration:
                    filtration[sigma] = get_circumcenter(X[sigma, :])[1]
                for i in range(dim):
                    # Propagate alpha filtration value
                    tau = sigma[0:i] + sigma[i+1::]
                    if tau in filtration:
                        filtration[tau] = min(filtration[tau], filtration[sigma])
                    elif len(tau) > 1:
                        # If Tau is not empty
                        xtau, rtauSqr = get_circumcenter(X[tau, :])
                        if np.sum((X[sigma[i], :]-xtau)**2) < rtauSqr:
                            filtration[tau] = filtration[sigma]
    for f in filtration:
        filtration[f] = np.sqrt(filtration[f])

    ## Step 2: Take care of numerical artifacts that may result
    ## in simplices with greater filtration values than their co-faces
    for dim in range(maxdim+2, 2, -1):
        for sigma in simplices_bydim[dim]:
            for i in range(dim):
                tau = sigma[0:i] + sigma[i+1::]
                if filtration[tau] > filtration[sigma]:
                    filtration[tau] = filtration[sigma]
    if Verbose:
        print("Finished building alpha filtration (Elapsed Time %.3g)"%(time.time()-tic))
                    
    simplices = [([i], 0) for i in range(X.shape[0])]
    for tau in filtration:
        simplices.append((tau, filtration[tau]))
    return get_phat_dgms(simplices, Verbose=Verbose)

def rips__filtration_gudhi(D, p, coeff = 2, doPlot = False):
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

def alpha_filtration_gudhi(X):
    pass

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
    Is = rips_filtration(X, 1)
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
        Is2 = rips__filtration_gudhi(D, 1)
        plotDGM(Is2[1], color = 'r', marker = 'x')
    plt.title("H1")
    plt.show()

def testAlpha():
    from ripser import plot_dgms
    # Make a 3-sphere in 4 dimensions
    X = np.random.randn(200, 4)
    X = X/np.sqrt(np.sum(X**2, 1)[:, None])
    tic = time.time()
    dgms = alpha_filtration(X)
    phattime = time.time() - tic
    print("Elapsed Time: %.3g"%phattime)
    plot_dgms([dgms[i] for i in range(len(dgms))])
    plt.show()

def convertGUDHIPD(pers, dim):
    Is = []
    for i in range(dim):
        Is.append([])
    for i in range(len(pers)):
        (dim, (b, d)) = pers[i]
        Is[dim].append([b, d])
    #Put onto diameter scale so it matches rips more closely
    return [np.sqrt(np.array(I)) for I in Is]

def compareAlpha():
    import gudhi
    np.random.seed(2)
    # Make a 4-sphere in 5 dimensions
    X = np.random.randn(100, 5)
    
    tic = time.time()
    Is1 = alpha_filtration(X)
    phattime = time.time() - tic
    print("Phat Time: %.3g"%phattime)
    
    tic = time.time()
    alpha_complex = gudhi.AlphaComplex(points = X.tolist())
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square = np.inf)
    pers = simplex_tree.persistence()
    gudhitime = time.time()-tic
    Is2 = convertGUDHIPD(pers, len(Is1))   
    
    print("GUDHI Time: %.3g"%gudhitime)
    
    I1 = Is1[len(Is1)-1]
    I2 = Is2[len(Is2)-1]
    plt.scatter(I1[:, 0], I1[:, 1])
    plt.scatter(I2[:, 0], I2[:, 1], 40, marker='x')
    plt.show()
    

if __name__ == '__main__':
    testRips()
    #testAlpha()
    #compareAlpha()

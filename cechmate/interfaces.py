def rips__filtration_gudhi(D, p, coeff=2, doPlot=False):
    """
    Do the rips filtration, wrapping around the GUDHI library (for comparison)
    :param X: An Nxk matrix of points
    :param p: The order of homology to go up to
    :param coeff: The field coefficient of homology
    :returns Is: A dictionary of persistence diagrams, where Is[k] is \
        the persistence diagram for Hk 
    """
    import gudhi

    rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=np.inf)
    simplex_tree = rips.create_simplex_tree(max_dimension=p + 1)
    diag = simplex_tree.persistence(homology_coeff_field=coeff, min_persistence=0)
    if doPlot:
        pplot = gudhi.plot_persistence_diagram(diag)
        pplot.show()
    Is = []
    for i in range(p + 1):
        Is.append([])
    for (i, (b, d)) in diag:
        Is[i].append([b, d])
    for i in range(len(Is)):
        Is[i] = np.array(Is[i])
    return Is


def convertGUDHIPD(pers, dim):
    Is = []
    for i in range(dim):
        Is.append([])
    for i in range(len(pers)):
        (dim, (b, d)) = pers[i]
        Is[dim].append([b, d])
    # Put onto diameter scale so it matches rips more closely
    return [np.sqrt(np.array(I)) for I in Is]


def compareAlpha():
    import gudhi

    np.random.seed(2)
    # Make a 4-sphere in 5 dimensions
    X = np.random.randn(100, 5)

    tic = time.time()
    Is1 = alpha_filtration(X)
    phattime = time.time() - tic
    print("Phat Time: %.3g" % phattime)

    tic = time.time()
    alpha_complex = gudhi.AlphaComplex(points=X.tolist())
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=np.inf)
    pers = simplex_tree.persistence()
    gudhitime = time.time() - tic
    Is2 = convertGUDHIPD(pers, len(Is1))

    print("GUDHI Time: %.3g" % gudhitime)

    I1 = Is1[len(Is1) - 1]
    I2 = Is2[len(Is2) - 1]
    plt.scatter(I1[:, 0], I1[:, 1])
    plt.scatter(I2[:, 0], I2[:, 1], 40, marker="x")
    plt.show()

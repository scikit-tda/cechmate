"""All filtrations should have a base interface.

"""

from ..solver import phat_diagrams


class BaseFiltration:
    """Base filtration that implements constructor and `diagrams` method.
    """

    def __init__(self, maxdim=None, verbose=True):
        """Default constructor
        
        Parameters
        ----------

        maxdim: int
            Maximum dimension of homology to compute
        verbose: boolean
            If True, then print logging statements.

        """

        self.maxdim = maxdim
        self.verbose = verbose

        self.simplices_ = None
        self.diagrams_ = None

    def diagrams(self, simplices=None, show_inf=False):
        """Compute persistence diagrams for the simplices.

        Parameters
        -----------
        simplices: 
            simplices or filtration built from :code:`build` method.

        show_inf: Boolean
            Determines whether or not to return points that never die.

        Returns
        ---------
        dgms: list of diagrams 
            the persistence diagram for Hk 

        """
        simplices = simplices or self.simplices_
        self.diagrams_ = phat_diagrams(simplices, show_inf)

        return self.diagrams_

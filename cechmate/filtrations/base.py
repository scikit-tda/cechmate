"""

All filtrations should have a base interface.



"""

from ..solver import phat_diagrams


class BaseFiltration:
    """Base filtration that implements constructor and `diagrams` method.
    """
 
    def __init__(self, max_dim=3, verbose=True):
        """Default constructor
        
        Parameters
        ----------

        max_dim: int
            Maximum dimension of simplices returned in constructed filtration.
        verbose: boolean
            If True, then print logging statements.

        """


        self.max_dim = max_dim
        self.verbose = verbose

        self.simplices_ = None
        self.diagrams_ = None

    def diagrams(self, simplices=None):
        """ Compute persistence diagrams for the simplices.

        Parameters
        -----------

        simplices: 
            simplices or filtration built from :code:`build` method.

        """
        simplices = simplices or self.simplices_
        self.diagrams_ = phat_diagrams(simplices)

        return self.diagrams_


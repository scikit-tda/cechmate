"""

All filtrations should have a base interface.



"""

from ..solver import phat_diagrams


class BaseFiltration:
    """Base filtration that implements constructor and `diagrams` method.

    """


    def __init__(self, max_dim=3, verbose=True):
        """This init should show up inherited.
        """
        self.max_dim = max_dim
        self.verbose = verbose

        self.simplices_ = None
        self.diagrams_ = None

    def diagrams(self, simplices=None):
        """ Compute persistence diagrams for the simplices.

        """
        simplices = simplices or self.simplices_
        self.diagrams_ = phat_diagrams(simplices)

        return self.diagrams_


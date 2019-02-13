"""

All filtrations should have a base interface.



"""

from ..solver import phat_diagrams


class BaseFiltration:
    def __init__(self, max_dim=3):
        self.max_dim = max_dim

        self.simplices_ = None
        self.diagrams_ = None

    def diagrams(self, simplices=None):
        simplices = simplices or self.simplices_
        self.diagrams_ = phat_diagrams(simplices)

        return self.diagrams_


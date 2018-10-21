"""

All filtrations should have a base interface.



"""


class BaseFiltration:
    def __init__(self, max_dim=3):
        self.max_dim = max_dim


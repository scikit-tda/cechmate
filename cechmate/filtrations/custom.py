import itertools
import time


from .base import BaseFiltration

__all__ = ["Custom"]


class Custom(BaseFiltration):
    def __init__(self):
        self.simplices_ = None

    def build(self, simplices):
        """
        OOP interface for custom filtration construction. Supply the filtration in the form of a list of simplices. Then construct diagrams with :code:`.diagrams` method.
        
        Parameters
        ===========
        simplices: List[tuple(float, List)]
            List of simplices as pairs of 
            
        """

        self.simplices_ = simplices

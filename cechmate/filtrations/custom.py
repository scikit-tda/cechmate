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

    def diagrams(self, simplices=None, show_inf=False, verbose=True, simplicial=True):
        """
        Redefine the Custom version of diagrams so that simplicial defaults to True, rather
        than false.
        """

        return super().diagrams(simplices,show_inf,verbose,simplicial)
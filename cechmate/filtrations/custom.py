import warnings

from cechmate.filtrations.base import BaseFiltration

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
        warnings.warn(
            "This method is deprecated and will be removed in future versions. Use the `fit` method instead.",
            DeprecationWarning,
        )

        self.simplices_ = simplices

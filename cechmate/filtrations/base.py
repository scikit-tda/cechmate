"""All filtrations should have a base interface."""

from typing import Literal
import warnings
from numpy.typing import NDArray
import numpy as np

# from ..solver import phat_diagrams


class BaseFiltration:
    """Base filtration that implements constructor and `diagrams` method."""

    def __init__(
        self,
        maxdim=None,
        verbose=True,
        solver: Literal["phat", "gudhi", "ripser"] = "gudhi",
    ):
        """Base filtration class.

        Not all solvers are implemented for all filtrations. See each filtration class for details.

        Parameters
        ----------

        maxdim: int
            Maximum dimension of homology to compute
        verbose: boolean
            If True, then print logging statements.
        solver: Literal["phat", "gudhi", "ripser"], default="gudhi"
            Solver to use for persistent homology.
        """

        self.maxdim = maxdim
        self.verbose = verbose
        self.solver = solver

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
        warnings.warn(
            "This function is deprecated and will be removed in a future release. Use transform instead.",
            DeprecationWarning,
        )
        simplices = simplices or self.simplices_
        # TODO: Update this call.
        self.diagrams_ = phat_diagrams(simplices, show_inf)

        return self.diagrams_

    def transform(self, simplices=None, ripser_format=True) -> list[NDArray]:
        """
        Compute persistent homology.
        """
        import gudhi

        simplices_ = simplices or self.simplices_

        if simplices_ is None:
            raise ValueError("No simplices to transform.")

        simplex_tree = gudhi.SimplexTree()
        for simplex, filtration_value in simplices_:
            simplex_tree.insert(simplex, filtration_value)

        persistence = simplex_tree.persistence()

        if not ripser_format:
            return persistence

        # convert to ripser.py format
        ripser_output = []
        for dim, (birth, death) in persistence:
            while len(ripser_output) <= dim:
                ripser_output.append([])
            if death == float("inf"):
                death = -1
            ripser_output[dim].append(np.array([birth, death]))
        ripser_output = [np.array(dgm) for dgm in ripser_output]

        return ripser_output

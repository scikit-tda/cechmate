import numpy as np
import matplotlib.pyplot as plt


def plot_diagram(
    diagram,
    color="b",
    sz=20,
    label="diagram",
    axcolor=np.array([0.0, 0.0, 0.0]),
    marker=None,
):
    """
    Plot a persistence diagram
    :param diagram: An NPoints x 2 array of birth and death times
    :param color: A color for the points (default 'b' for blue)
    :param sz: Size of the points
    :param label: Label to associate with the diagram
    :param axcolor: Color of the diagonal
    :param marker: Type of marker (e.g. 'x' for an x marker)
    :returns H: A handle to the plot
    """
    if diagram.size == 0:
        return
    # Create Lists
    # set axis values
    axMin = np.min(diagram)
    axMax = np.max(diagram)
    axRange = axMax - axMin
    a = max(axMin - axRange / 5, 0)
    b = axMax + axRange / 5
    # plot line
    plt.plot([a, b], [a, b], c=axcolor, label="none")
    # plot points
    if marker:
        H = plt.scatter(
            diagram[:, 0],
            diagram[:, 1],
            sz,
            color,
            marker,
            label=label,
            edgecolor="none",
        )
    else:
        H = plt.scatter(
            diagram[:, 0], diagram[:, 1], sz, color, label=label, edgecolor="none"
        )
    # add labels
    plt.xlabel("Time of Birth")
    plt.ylabel("Time of Death")
    return H


__all__ = ["plot_diagram"]
